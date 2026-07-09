"""ImplantScan backend — marker pose detection via two-stage engraving-first ICP.

v3 pipeline (replaces DBSCAN cluster isolation entirely):
  Stage 1 — DETECT on the engraving:
    * crop scan tightly (r=7mm) around the click — the click is on the engraved
      face, so contamination from gum / neighbouring markers is minimal
    * register the SCAN CROP (source) into the TOP-SLAB template (target: top
      face + engraving, top 2.5mm). Registering crop->template keeps fitness
      meaningful even when the crop covers only part of a long marker.
    * validate each candidate pose:
        - orientation: template +Z must agree with the scan surface normal at
          the click (kills flips and squashed degenerate poses)
        - coverage: slab template points (within the crop sphere) must land on
          real scan surface
  Stage 2 — REFINE on the full visible surface:
    * transform scan points into template frame, keep only points inside the
      template's actual bounding box (+0.4mm) — a geometric cut that removes
      gum and neighbouring markers exactly, no clustering needed
    * point-to-point ICP of boxed scan -> visible template, inverted at the end

Why: real intraoral scans have markers touching each other and fused to gum.
Anything based on 'isolate a cluster' fails there; this pipeline never needs
isolation. Validated on a real Aoralscan Elf upper-jaw scan with L+M markers
physically touching and S fused to gum: 27/30 random clicks accepted, 0 wrong
poses returned, accepted solutions repeatable to ~45um, ~1.3s per request.

Deploy on Railway. Endpoints:
  GET  /            health check
  POST /register    body: {scan:{vertices[,faces]}, template:{vertices,faces},
                           click:[x,y,z], marker_len: float,
                           view_dir:[x,y,z] (optional, camera->click direction)}
                    returns: {ok, reliable, matrix 4x4, rms, fitness}
"""
import numpy as np
import open3d as o3d
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from scipy.spatial import cKDTree

app = FastAPI(title="ImplantScan ICP")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

VOXEL = 0.35            # tuned for Aoralscan Elf density (mean edge ~0.18mm)
CROP_R = 7.0            # stage-1 crop radius around the click (mm)
SLAB_DEPTH = 2.5        # engraving slab thickness taken from template top (mm)
MIN_STAGE1_FITNESS = 0.45
MIN_COVERAGE = 0.55     # slab-template coverage by scan points
MIN_FITNESS = 0.85      # final acceptance — below this the pose may be a
                        # lengthwise alias of the engraving; reject, re-click
TRIALS = 10

class Mesh(BaseModel):
    vertices: List[List[float]]
    faces: Optional[List[List[int]]] = None

class RegisterReq(BaseModel):
    scan: Mesh
    template: Mesh
    click: List[float]
    marker_len: float
    view_dir: Optional[List[float]] = None   # camera->surface direction at click

def sample_template(mesh: Mesh, n_points=30000):
    """Returns (top_slab_pcd, visible_pcd, visible_pts, bounds_lo, bounds_hi).

    top slab  = top face + engraving walls/floor (the unique fingerprint —
                the only part guaranteed visible in every scan)
    visible   = everything except the mounting face / underside
    bounds    = the template's REAL bounding box (templates are exported
                side-by-side in Fusion space and are NOT centred at origin)
    """
    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int32)
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    m.compute_vertex_normals()
    p = m.sample_points_uniformly(number_of_points=n_points)
    pts, nrm = np.asarray(p.points), np.asarray(p.normals)
    zmax = pts[:, 2].max()
    top = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(pts[pts[:, 2] > zmax - SLAB_DEPTH]))
    keep = nrm[:, 2] > -0.3
    vis_pts = pts[keep]
    vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vis_pts))
    return top, vis, vis_pts.min(0) - 0.4, vis_pts.max(0) + 0.4

def prep(p, voxel=VOXEL):
    pd = p.voxel_down_sample(voxel)
    pd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2, max_nn=30))
    fp = o3d.pipelines.registration.compute_fpfh_feature(
        pd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5, max_nn=100))
    return pd, fp

def click_normal(scan_pts, click, view_dir=None, k=60):
    """Surface normal at the click, from local PCA. Sign: against the camera
    ray if the frontend sent one; otherwise sign is resolved later by |dot|."""
    tree = cKDTree(scan_pts)
    _, idx = tree.query(click, k=min(k, len(scan_pts)))
    q = scan_pts[idx] - scan_pts[idx].mean(0)
    n = np.linalg.svd(q, full_matrices=False)[2][2]
    n /= np.linalg.norm(n)
    if view_dir is not None:
        vd = np.asarray(view_dir, float)
        if np.dot(n, vd) > 0:            # normal must face the camera
            n = -n
        return n, True
    return n, False

def register(scan_pts, template: Mesh, click, marker_len, view_dir=None):
    click = np.asarray(click, float)
    top, vis, box_lo, box_hi = sample_template(template)
    top_pts = np.asarray(top.points)

    d = np.linalg.norm(scan_pts - click, axis=1)
    crop_pts = scan_pts[d < CROP_R]
    if len(crop_pts) < 100:
        return None
    crop = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(crop_pts))
    n_click, strict_sign = click_normal(scan_pts, click, view_dir)

    sd, sf = prep(crop)       # source = scan crop
    td, tf = prep(top)        # target = engraving slab
    dist = VOXEL * 1.5
    crop_tree = cKDTree(crop_pts)

    best, best_score = None, -1.0
    for _ in range(TRIALS):
        res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            sd, td, sf, tf, True, dist,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(300000, 0.9999))
        icp = o3d.pipelines.registration.registration_icp(
            sd, td, VOXEL*2, res.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
        if icp.fitness < MIN_STAGE1_FITNESS:
            continue
        T = np.linalg.inv(icp.transformation)        # template -> world
        z_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        agree = np.dot(z_world, n_click)
        if (strict_sign and agree < 0.7) or (not strict_sign and abs(agree) < 0.7):
            continue
        tw = (T[:3, :3] @ top_pts.T).T + T[:3, 3]
        inside = np.linalg.norm(tw - click, axis=1) < CROP_R
        if inside.sum() < 200:
            continue
        dd, _ = crop_tree.query(tw[inside], k=1)
        cov = (dd < VOXEL*2).mean()
        if cov < MIN_COVERAGE:
            continue
        score = cov + icp.fitness
        if score > best_score:
            best_score, best = score, T
        if cov > 0.8 and icp.fitness > 0.8:
            break
    if best is None:
        return None

    # ---- stage 2: pose-guided box crop + refine ----
    big = scan_pts[np.linalg.norm(scan_pts - click, axis=1) < marker_len*0.7 + 3]
    inv = np.linalg.inv(best)
    local = (inv[:3, :3] @ big.T).T + inv[:3, 3]
    inbox = np.all((local > box_lo) & (local < box_hi), axis=1)
    boxed_pts = big[inbox]
    if len(boxed_pts) < 200:
        return None
    boxed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(boxed_pts))

    def refine(init_inv):
        r = o3d.pipelines.registration.registration_icp(
            boxed, vis, 0.5, init_inv,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=150))
        return np.linalg.inv(r.transformation), r

    # ---- flip disambiguation on the ENGRAVING ----
    # The smooth marker body is 180deg-symmetric about its vertical axis, so a
    # yaw-flipped pose can still score fitness >= 0.85 (body matches body; only
    # the engraving disagrees, and it is a minority of points). Refine BOTH the
    # found pose and its flipped twin, score them on slab-template points with
    # a tight 0.25mm threshold, keep the winner, and reject if the twins are
    # too close to call.
    slab_ctr = (top_pts.min(0) + top_pts.max(0)) / 2.0
    Rz = np.diag([-1.0, -1.0, 1.0])            # 180deg about vertical thru slab centre
    T_a, fine_a = refine(inv)
    T_b0 = best.copy()
    T_b0[:3, :3] = best[:3, :3] @ Rz
    T_b0[:3, 3] = best[:3, 3] + best[:3, :3] @ (slab_ctr - Rz @ slab_ctr)
    T_b, fine_b = refine(np.linalg.inv(T_b0))

    scan_tree = cKDTree(scan_pts)
    def engraving_frac(T):
        w = (np.asarray(T)[:3, :3] @ top_pts.T).T + np.asarray(T)[:3, 3]
        dd, _ = scan_tree.query(w, k=1)
        return (dd < 0.25).mean()

    ea, eb = engraving_frac(T_a), engraving_frac(T_b)
    if ea >= eb:
        T, fine, e_win, e_lose = T_a, fine_a, ea, eb
    else:
        T, fine, e_win, e_lose = T_b, fine_b, eb, ea
    engraving_ok = (e_win >= 0.72) and (e_win - e_lose >= 0.06)
    return T, fine.fitness, fine.inlier_rmse, engraving_ok

@app.get("/")
def health():
    return {"status": "ok", "service": "ImplantScan ICP", "pipeline": "v3 engraving-first"}

@app.post("/register")
def register_endpoint(req: RegisterReq):
    scan_v = np.asarray(req.scan.vertices, dtype=float)
    out = register(scan_v, req.template, req.click, req.marker_len, req.view_dir)
    if out is None:
        return {"ok": False,
                "error": "registration failed — click on the engraved face of the marker"}
    T, fitness, rmse, engraving_ok = out
    reliable = (fitness >= MIN_FITNESS) and engraving_ok
    return {"ok": True,
            "reliable": bool(reliable),
            "matrix": [list(map(float, row)) for row in np.asarray(T)],
            "rms": float(rmse),
            "fitness": float(fitness)}
