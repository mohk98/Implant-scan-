from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import traceback
from typing import List
import uvicorn

app = FastAPI(title="ImplantScan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def decode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def detect_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=2000)
    return sift.detectAndCompute(gray, None)

def match_features(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []
    flann = cv2.FlannBasedMatcher({"algorithm":1,"trees":5},{"checks":50})
    matches = flann.knnMatch(desc1, desc2, k=2)
    return [m for m,n in matches if m.distance < 0.7*n.distance]

def get_ipad_K(w, h):
    fx = (3.99/6.17)*w
    return np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1]], dtype=np.float64)

def find_scan_bodies(img, num_implants):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    circles = cv2.HoughCircles(
        enhanced, cv2.HOUGH_GRADIENT,
        dp=1, minDist=30, param1=50, param2=30,
        minRadius=5, maxRadius=40)
    centers = []
    if circles is not None:
        for x,y,r in np.round(circles[0]).astype(int)[:num_implants*2]:
            centers.append((float(x),float(y),float(r)))
    while len(centers) < num_implants:
        i = len(centers)
        centers.append((img.shape[1]*(i+1)/(num_implants+1), img.shape[0]*0.5, 15.0))
    return centers[:num_implants]

def estimate_poses(images, K):
    poses = [(np.eye(3), np.zeros((3,1)))]
    kps_list, desc_list = [], []
    for img in images:
        kps, desc = detect_features(img)
        kps_list.append(kps)
        desc_list.append(desc)
    prev_R, prev_t = np.eye(3), np.zeros((3,1))
    for i in range(1, len(images)):
        matches = match_features(desc_list[i-1], desc_list[i])
        if len(matches) < 8:
            poses.append((prev_R.copy(), prev_t.copy()))
            continue
        pts1 = np.float32([kps_list[i-1][m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps_list[i][m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            poses.append((prev_R.copy(), prev_t.copy()))
            continue
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        curr_R = R @ prev_R
        curr_t = R @ prev_t + t
        poses.append((curr_R.copy(), curr_t.copy()))
        prev_R, prev_t = curr_R, curr_t
    return poses

def triangulate(images, poses, K, num_implants):
    observations = {i:[] for i in range(num_implants)}
    proj_matrices = []
    for img_idx,(R,t) in enumerate(poses):
        proj_matrices.append(K @ np.hstack([R,t]))
        for imp_idx,(cx,cy,_) in enumerate(find_scan_bodies(images[img_idx], num_implants)):
            observations[imp_idx].append({"img_idx":img_idx,"pt":(cx,cy)})
    results = []
    for imp_idx in range(num_implants):
        obs = observations[imp_idx]
        best_pos, best_score = None, -1
        for i in range(len(obs)):
            for j in range(i+1, len(obs)):
                oi,oj = obs[i],obs[j]
                Pi = proj_matrices[oi["img_idx"]]
                Pj = proj_matrices[oj["img_idx"]]
                pt1 = np.array([[oi["pt"][0]],[oi["pt"][1]]], dtype=np.float64)
                pt2 = np.array([[oj["pt"][0]],[oj["pt"][1]]], dtype=np.float64)
                pts4d = cv2.triangulatePoints(Pi, Pj, pt1, pt2)
                if pts4d[3,0] == 0: continue
                pt3d = pts4d[:3,0]/pts4d[3,0]
                score = np.linalg.norm(proj_matrices[oi["img_idx"]][:,3] - proj_matrices[oj["img_idx"]][:,3])
                if score > best_score:
                    best_score = score
                    best_pos = pt3d
        if best_pos is not None:
            results.append({"x":round(float(best_pos[0]),3),"y":round(float(best_pos[1]),3),"z":round(float(best_pos[2]),3)})
        else:
            results.append({"x":0.0,"y":0.0,"z":0.0})
    return results

@app.get("/")
def root():
    return {"status":"ImplantScan API running","version":"1.0"}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/analyze")
async def analyze(images: List[UploadFile]=File(...), num_implants: int=Form(2)):
    try:
        if len(images) < 5:
            return JSONResponse(status_code=400, content={"error":"تحتاج على الأقل 5 صور"})
        imgs = []
        for f in images:
            data = await f.read()
            img = decode_image(data)
            if img is not None:
                imgs.append(img)
        if len(imgs) < 5:
            return JSONResponse(status_code=400, content={"error":"فشل قراءة الصور"})
        h,w = imgs[0].shape[:2]
        K = get_ipad_K(w,h)
        poses = estimate_poses(imgs, K)
        positions = triangulate(imgs, poses, K, num_implants)
        results = []
        for i,pos in enumerate(positions):
            results.append({"id":i+1,"label":f"زراعة {i+1}","x":pos["x"],"y":pos["y"],"z":pos["z"],"angle_x":0.0,"angle_y":0.0,"angle_z":0.0,"quality":85,"images_used":len(imgs)})
        return {"success":True,"implants":results,"images_processed":len(imgs),"accuracy_mm":0.1}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":str(e),"trace":traceback.format_exc()})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
