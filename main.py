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

def to_pcd(mesh: Mesh, n_points=8000):
    v = np.asarray(mesh.vertices, dtype=float)
    if mesh.faces:
        f = np.asarray(mesh.faces, dtype=np.int32)
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
        return m.sample_points_uniformly(number_of_points=n_points)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v))

def crop_sphere(pcd, center, radius):
    pts = np.asarray(pcd.points)
    keep = np.linalg.norm(pts - center, axis=1) <= radius
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[keep]))

def register(template_pcd, scan_pcd, click, marker_len, voxel=0.35):
    target = crop_sphere(scan_pcd, np.array(click, float), marker_len*0.75)
    if len(target.points) < 30:
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
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        sd, td, sf, tf, True, dist,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2, max_nn=30))
    icp = o3d.pipelines.registration.registration_icp(
        src, target, voxel*2, res.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
    return icp

@app.get("/")
def health():
    return {"status": "ok", "service": "ImplantScan ICP"}

@app.post("/register")
def register_endpoint(req: RegisterReq):
    tpl = to_pcd(req.template, 8000)
    scan = to_pcd(req.scan, 40000)
    icp = register(tpl, scan, req.click, req.marker_len)
    if icp is None:
        return {"ok": False, "error": "not enough points near click"}
    T = icp.transformation
    return {"ok": True,
            "matrix": [list(map(float, row)) for row in T],
            "rms": float(icp.inlier_rmse),
            "fitness": float(icp.fitness)}
