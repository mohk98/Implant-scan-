"""ImplantScan backend — marker pose detection via open3d ICP.
Deploy on Railway. Endpoints:
  GET  /            health check
  POST /register    body: {scan: {vertices, faces}, template: {vertices, faces},
                           click: [x,y,z], marker_len: float}
                    returns: {matrix: 4x4, rms, fitness}
"""
import numpy as np
import open3d as o3d
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="ImplantScan ICP")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class Mesh(BaseModel):
    vertices: List[List[float]]
    faces: Optional[List[List[int]]] = None

class RegisterReq(BaseModel):
    scan: Mesh
    template: Mesh
    click: List[float]
    marker_len: float

def to_pcd(mesh: Mesh, n_points=20000):
    v = np.asarray(mesh.vertices, dtype=float)
    if mesh.faces:
        f = np.asarray(mesh.faces, dtype=np.int32)
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
        m.compute_vertex_normals()
        return m.sample_points_uniformly(number_of_points=n_points)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v))

def visible_template(mesh: Mesh, n_points=20000, drop_thresh=-0.3):
    """Sample the template but DROP downward-facing surfaces.

    The intraoral scanner never sees the marker's mounting face (it's glued to
    the implant/cast) nor the inside of the screw hole. Registering a full
    closed template against a partially-observed surface is what made RANSAC
    fail ~90% of the time. Keeping only the outward/upward-facing surface makes
    registration repeatable.
    """
    v = np.asarray(mesh.vertices, dtype=float)
    if not mesh.faces:
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v))
    f = np.asarray(mesh.faces, dtype=np.int32)
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    m.compute_vertex_normals()
    p = m.sample_points_uniformly(number_of_points=n_points)
    pts = np.asarray(p.points); nrm = np.asarray(p.normals)
    keep = nrm[:, 2] > drop_thresh
    if keep.sum() < 100:
        return p
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[keep]))

def crop_sphere(pcd, center, radius):
    pts = np.asarray(pcd.points)
    keep = np.linalg.norm(pts - center, axis=1) <= radius
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[keep]))

def isolate_cluster(pcd, click, eps, min_pts=10):
    """Keep only the connected cluster nearest the click — removes neighbouring
    markers / cast surface that would confuse registration."""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_pts))
    if labels.max() < 0:
        return pcd
    pts = np.asarray(pcd.points)
    best, bd = -1, 1e9
    for lab in range(labels.max()+1):
        d = np.linalg.norm(pts[labels==lab] - np.array(click), axis=1).min()
        if d < bd:
            bd, best = d, lab
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[labels==best]))

def register(template_pcd, scan_pcd, click, marker_len, voxel=0.4, trials=12):
    """Tuned on a real Aoralscan Elf export (mean edge ~0.18mm)."""
    click = np.array(click, float)
    target = crop_sphere(scan_pcd, click, marker_len*0.7)
    if len(target.points) < 50:
        return None
    target = isolate_cluster(target, click, eps=voxel*2.5)
    if len(target.points) < 50:
        return None
    src = template_pcd

    def prep(p):
        pd = p.voxel_down_sample(voxel)
        pd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2, max_nn=30))
        fp = o3d.pipelines.registration.compute_fpfh_feature(
            pd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5, max_nn=100))
        return pd, fp

    sd, sf = prep(src); td, tf = prep(target)
    dist = voxel*1.5
    best = None
    MIN_FITNESS = 0.60          # below this the pose is unreliable — keep trying
    for _ in range(trials):
        res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            sd, td, sf, tf, True, dist,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 0.9999))
        icp = o3d.pipelines.registration.registration_icp(
            sd, td, voxel*2, res.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=120))
        if icp.fitness < 0.15:
            continue
        centre_dist = np.linalg.norm(icp.transformation[:3,3] - click)
        score = icp.inlier_rmse - icp.fitness + max(0, centre_dist-4)*0.5
        if best is None or score < best[0]:
            best = (score, icp)
        # good enough — stop early
        if icp.fitness >= 0.70 and centre_dist < 4:
            break
    if best is None:
        return None
    icp = best[1]
    # flag unreliable results instead of silently returning a wrong pose
    icp_reliable = icp.fitness >= MIN_FITNESS
    return icp, icp_reliable

@app.get("/")
def health():
    return {"status": "ok", "service": "ImplantScan ICP"}

@app.post("/register")
def register_endpoint(req: RegisterReq):
    tpl = visible_template(req.template, 20000)
    scan = to_pcd(req.scan, 60000)
    out = register(tpl, scan, req.click, req.marker_len)
    if out is None:
        return {"ok": False, "error": "registration failed — click nearer the marker"}
    icp, reliable = out
    T = icp.transformation
    return {"ok": True,
            "reliable": bool(reliable),
            "matrix": [list(map(float, row)) for row in T],
            "rms": float(icp.inlier_rmse),
            "fitness": float(icp.fitness)}
