"""
ImplantScan Backend v2 — Multi-View Triangulation
==================================================
بدل الاعتماد على صورة وحدة (monocular pose)، نستخدم عدة إطارات (frames)
من نفس الماركرات الثابتة، ونحسب المواضع النسبية بينها ثم نعمل
robust averaging عبر كل الإطارات لتقليل الخطأ العشوائي بشكل كبير.

الفكرة الرياضية:
  لكل frame نحسب pose كل ماركر بالنسبة للكاميرا (solvePnP)
  نختار ماركر مرجعي (reference) — عادة أقل ID موجود بأغلب الـ frames
  نحسب التحويل من المرجع لكل ماركر ثاني بكل frame:
      T_ref_to_B = inverse(T_cam_to_ref) @ T_cam_to_B
  هذا يلغي خطأ "بعد الكاميرا عن الماركر" لأنه نسبي بين ماركرين بنفس الصورة
  ثم نجمع كل التقديرات عبر كل الـ frames ونأخذ median/trimmed-mean
  هذا يقلل الخطأ العشوائي بمعامل تقريبي 1/sqrt(N) عدد الإطارات
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import cv2

app = FastAPI(title="ImplantScan Backend v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════ Data Models ═══════════════

class Corner(BaseModel):
    x: float
    y: float


class MarkerDetection(BaseModel):
    id: int
    corners: List[Corner]   # 4 corners، بترتيب [TL, TR, BR, BL] (نفس ترتيب js-aruco2)


class FrameData(BaseModel):
    width: int
    height: int
    markers: List[MarkerDetection]


class TriangulateRequest(BaseModel):
    frames: List[FrameData]
    focal_length_px: float     # من الكالبريشن اللي التطبيق سواه
    marker_size_mm: float = 8.0


class MarkerResult(BaseModel):
    id: int
    x: float
    y: float
    z: float
    num_observations: int
    std_dev_mm: float          # مؤشر عدم اليقين — كل ما قل كان أدق


class TriangulateResponse(BaseModel):
    success: bool
    reference_marker_id: Optional[int] = None
    markers: List[MarkerResult]
    total_frames_used: int
    message: str = ""


# ═══════════════ Core Math ═══════════════

def build_object_points(size_mm: float) -> np.ndarray:
    """
    نقاط الزوايا الأربعة للماركر بإحداثياته المحلية الخاصة (z=0)
    نفس ترتيب js-aruco2: [TL, TR, BR, BL]
    """
    h = size_mm / 2.0
    return np.array([
        [-h,  h, 0],   # TL
        [ h,  h, 0],   # TR
        [ h, -h, 0],   # BR
        [-h, -h, 0],   # BL
    ], dtype=np.float64)


def solve_marker_pose(corners_px: np.ndarray, camera_matrix: np.ndarray,
                       size_mm: float) -> Optional[np.ndarray]:
    """
    يحسب pose الماركر بالنسبة للكاميرا (transform 4x4: camera → marker)
    يرجع None لو فشل الحساب
    """
    obj_pts = build_object_points(size_mm)
    dist_coeffs = np.zeros((4, 1))  # نفترض بدون distortion كبير (كاميرات الآيباد الحديثة جيدة)

    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, corners_px, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE  # مخصص لمربعات — أدق وأسرع
        )
    except cv2.error:
        ok = False

    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def trimmed_mean(values: np.ndarray, trim_fraction: float = 0.2) -> np.ndarray:
    """
    نرفض أسوأ trim_fraction من القيم (الأبعد عن المتوسط) قبل الحساب النهائي
    هذا يحمي من outliers الناتجة عن كشف زوايا خاطئ بإطار معين
    """
    if len(values) <= 3:
        return np.median(values, axis=0)

    center = np.median(values, axis=0)
    dists = np.linalg.norm(values - center, axis=1)
    n_keep = max(3, int(len(values) * (1 - trim_fraction)))
    keep_idx = np.argsort(dists)[:n_keep]
    return np.mean(values[keep_idx], axis=0)


# ═══════════════ Main Endpoint ═══════════════

@app.post("/triangulate", response_model=TriangulateResponse)
def triangulate(req: TriangulateRequest):
    if not req.frames:
        raise HTTPException(400, "لا توجد frames للمعالجة")

    # نحسب pose كل ماركر بكل frame
    # per_frame_poses[frame_idx] = { marker_id: T_cam_to_marker }
    per_frame_poses: List[Dict[int, np.ndarray]] = []

    for frame in req.frames:
        cx, cy = frame.width / 2.0, frame.height / 2.0
        camera_matrix = np.array([
            [req.focal_length_px, 0, cx],
            [0, req.focal_length_px, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        frame_poses: Dict[int, np.ndarray] = {}
        for m in frame.markers:
            if len(m.corners) != 4:
                continue
            corners_px = np.array(
                [[c.x, c.y] for c in m.corners], dtype=np.float64
            )
            T = solve_marker_pose(corners_px, camera_matrix, req.marker_size_mm)
            if T is not None:
                frame_poses[m.id] = T

        if frame_poses:
            per_frame_poses.append(frame_poses)

    if not per_frame_poses:
        return TriangulateResponse(
            success=False, markers=[], total_frames_used=0,
            message="ما قدرنا نحسب pose لأي ماركر بأي frame"
        )

    # نختار الماركر المرجعي = الأكثر ظهوراً عبر كل الـ frames
    id_counts: Dict[int, int] = {}
    for fp in per_frame_poses:
        for mid in fp:
            id_counts[mid] = id_counts.get(mid, 0) + 1

    ref_id = max(id_counts, key=id_counts.get)

    # لكل ماركر ثاني، نجمع كل تقديرات T_ref_to_marker عبر الـ frames
    relative_translations: Dict[int, List[np.ndarray]] = {}

    for fp in per_frame_poses:
        if ref_id not in fp:
            continue
        T_cam_ref = fp[ref_id]
        T_ref_cam = invert_transform(T_cam_ref)

        for mid, T_cam_m in fp.items():
            if mid == ref_id:
                continue
            T_ref_to_m = T_ref_cam @ T_cam_m
            pos = T_ref_to_m[:3, 3]
            relative_translations.setdefault(mid, []).append(pos)

    # المرجع نفسه بموضع (0,0,0)
    results: List[MarkerResult] = [
        MarkerResult(id=ref_id, x=0.0, y=0.0, z=0.0,
                     num_observations=id_counts[ref_id], std_dev_mm=0.0)
    ]

    for mid, positions in relative_translations.items():
        arr = np.array(positions)
        final_pos = trimmed_mean(arr, trim_fraction=0.2)
        std_dev = float(np.std(np.linalg.norm(arr - final_pos, axis=1)))

        results.append(MarkerResult(
            id=mid,
            x=float(final_pos[0]),
            y=float(final_pos[1]),
            z=float(final_pos[2]),
            num_observations=len(positions),
            std_dev_mm=round(std_dev, 4)
        ))

    results.sort(key=lambda r: r.id)

    return TriangulateResponse(
        success=True,
        reference_marker_id=ref_id,
        markers=results,
        total_frames_used=len(per_frame_poses),
        message=f"تم حساب {len(results)} ماركر من {len(per_frame_poses)} إطار"
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0-triangulation"}
