import os
import csv
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from siamese_traine import SiameseNet, preprocess_and_fix_length, evaluate_against_references, LIMB_GROUPS
import numpy as np
import pandas as pd

def compute_motion_energy(file_path: str) -> float:
    """Compute mean velocity magnitude from TSV; safely handle NaNs."""
    try:
        df = pd.read_csv(file_path, sep=r"\s+|\t+", engine="python")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            print(f"[energy] No numeric columns in {file_path}")
            return 0.0

        data = df[numeric_cols].to_numpy()
        data = np.nan_to_num(data, nan=0.0)
        vel = np.gradient(data, axis=0)
        vel = np.nan_to_num(vel, nan=0.0)
        energy = np.sqrt((vel ** 2).sum(axis=1)).mean()
        if np.isnan(energy) or np.isinf(energy):
            energy = 0.0
        return float(energy)
    except Exception as e:
        print(f"[energy] Could not compute for {file_path}: {e}")
        return 0.0

def energy_penalize(score, file_path, threshold=0.05, min_frames=300):
    energy = compute_motion_energy(file_path)
    df = pd.read_csv(file_path, sep=r"\s+|\t+", engine="python")
    frames = len(df)
    if energy < threshold or frames < min_frames:
        e_factor = max(energy / threshold, 0.08)
        l_factor = max(frames / min_frames, 0.3)
        factor = min(e_factor, l_factor)
        print(f"[energy] low={energy:.4f}, frames={frames} → scaling by {factor:.2f}")
        score *= factor
    return float(score)

# Load model once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SiameseNet(J=25, C=6, d_model=128).to(device)
model.load_state_dict(torch.load("runs/siamese_cv/fold1/best_model.pth", map_location=device))
model.eval()

hy_dir = r"C:\Users\ROG\dataset\HY-1"
hy_refs = [os.path.join(hy_dir, f) for f in os.listdir(hy_dir) if f.endswith(".tsv")]

app = FastAPI()

class AnalyzeRequest(BaseModel):
    email: str
    session_number: int
    file_path: str

class AnalyzeResponse(BaseModel):
    progress: float
    feedback: List[str]
    feedbackFilePath: str

def safe_email_to_filename(email: str) -> str:
    return email.replace("@", "_at_").replace(".", "_")

def segment_time_label(idx: int, total_seconds: float = 20.0, n_segments: int = 4) -> str:
    seg_len = total_seconds / max(n_segments, 1)
    start = idx * seg_len
    end = (idx + 1) * seg_len
    return f"{int(start)}–{int(end)}s"

def build_feedback(overall: float, segs: List[float], limbs: dict) -> List[str]:
    fb = []
    if overall >= 85:
        fb.append("Excellent overall consistency—keep reinforcing current technique.")
    elif overall >= 70:
        fb.append("Strong overall form. Focus on polishing weaker moments for even more consistency.")
    elif overall >= 55:
        fb.append("Decent foundation. Work on stability and timing to raise overall consistency.")
    else:
        fb.append("Consistency needs work—slow down, focus on posture and clean transitions.")

    limb_items = [(k, v) for k, v in limbs.items() if v is not None]
    if limb_items:
        limb_items.sort(key=lambda kv: kv[1])
        weakest_limb, weakest_score = limb_items[0]
        if weakest_score < 50:
            fb.append(f"{weakest_limb}: large variability detected—reduce unnecessary movement and keep stable alignment.")
        elif weakest_score < 65:
            fb.append(f"{weakest_limb}: moderate inconsistency—tighten control and maintain constant speed.")
        elif weakest_score < 75:
            fb.append(f"{weakest_limb}: small refinements needed—aim for smoother transitions.")
        if len(limb_items) >= 2 and limb_items[1][1] < 70:
            fb.append(f"{limb_items[1][0]}: also slightly behind—practice precise start/stop positions.")

    if segs:
        low_segments = [(i, s) for i, s in enumerate(segs) if s < 70]
        if low_segments:
            low_segments.sort(key=lambda t: t[1])
            for i, s in low_segments[:2]:
                fb.append(f"Segment {i+1} ({segment_time_label(i, total_seconds=20.0, n_segments=len(segs))}): consistency {s:.0f}%. Slow down and match your reference timing.")

    if not fb:
        fb.append("Great work—minor tweaks only. Keep practicing with steady tempo and clear positions.")

    return fb[:4]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    safe_email = safe_email_to_filename(req.email)
    expected_path = os.path.join(r"C:\Users\ROG\TSV", f"{safe_email}_session{req.session_number}.tsv")
    file_path = expected_path if os.path.exists(expected_path) else req.file_path

    overall, segs, limbs, mean_hy = evaluate_against_references([model], file_path, hy_refs)
    
    bd_dir = r"C:\Users\ROG\dataset\BD-1"
    bd_refs = [os.path.join(bd_dir, f) for f in os.listdir(bd_dir) if f.endswith(".tsv")]
    hy_sim = evaluate_against_references([model], file_path, hy_refs)[0]
    bd_sim = evaluate_against_references([model], file_path, bd_refs)[0]
    if bd_sim > hy_sim:
        overall = min(overall, 40.0)

    overall = energy_penalize(overall, file_path)
    energy = np.sqrt(np.var(pd.read_csv(file_path, sep=r"\s+|\t+", engine="python").select_dtypes(float).to_numpy()))
    if energy < 1e-3:
        overall = 0.0

    feedback = build_feedback(overall, segs, limbs)

    out_dir = r"C:\Users\ROG\feedback"
    os.makedirs(out_dir, exist_ok=True)
    feedback_file = os.path.join(out_dir, f"results_{safe_email}_session{req.session_number}.csv")
    with open(feedback_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["email", req.email])
        w.writerow(["session_number", req.session_number])
        w.writerow(["used_file_path", file_path])
        w.writerow(["mean_hy_self_sim_for_calibration", f"{mean_hy:.3f}"])
        w.writerow([])
        w.writerow(["overall_calibrated_percent", f"{overall:.2f}"])
        if segs:
            w.writerow(["segments"] + [f"{s:.2f}" for s in segs])
            w.writerow(["segment_windows"] + [segment_time_label(i, 20.0, len(segs)) for i in range(len(segs))])
        w.writerow([])
        w.writerow(["limb", "score"])
        for limb, score in limbs.items():
            w.writerow([limb, f"{score:.2f}" if score is not None else "NA"])
        w.writerow([])
        w.writerow(["feedback"])
        for line in feedback:
            w.writerow([line])

    progress = float(overall)

    return AnalyzeResponse(
        progress=progress,
        feedback=feedback,
        feedbackFilePath=feedback_file
    )
