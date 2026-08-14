# kata_scoring.py
import os
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# -----------------------------
# Config
# -----------------------------
TARGET_HZ = 60.0       
SMOOTH_WIN_SECS = 0.12 
SMOOTH_POLY = 3
DTW_RADIUS = 15        
EPS = 1e-8

MARKERS = [
    'BKHD','RFHD','LFHD','C7','NCK','RSHD','LSHD','RELB','LELB','RWRS','LWRS',
    'RIND','LIND','RPNK','LPNK','RFHP','RBHP','LFHP','LBHP','RKNE','LKNE','RFAK','RBAK','LFAK','LBAK'
]

PELVIS = ['RFHP','RBHP','LFHP','LBHP']   
NECK = 'NCK'                              
HIPS_PAIR = ('RFHP','LFHP')               

ANGLE_DEFS = [
    ('RELB','RSHD','RWRS'), 
    ('LELB','LSHD','LWRS'), 
    ('RKNE','RFHP','RFAK'), 
    ('LKNE','LFHP','LFAK'), 
]

LIMB_GROUPS = {
    'Right Arm': ['RSHD','RELB','RWRS','RIND','RPNK'],
    'Left Arm':  ['LSHD','LELB','LWRS','LIND','LPNK'],
    'Right Leg': ['RFHP','RKNE','RFAK','RBAK'],
    'Left Leg':  ['LFHP','LKNE','LFAK','LBAK'],
    'Torso/Head':['BKHD','RFHD','LFHD','C7','NCK','RBHP','LBHP']
}

# -----------------------------
# IO & parsing
# -----------------------------
def load_qualisys_tsv(file_path: str, marker_names: List[str], scale_mm_to_m=1000.0) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data_start = next(i for i, line in enumerate(lines) if line.strip().startswith('Frame'))
    df = pd.read_csv(file_path, sep='\t', skiprows=data_start, header=0)

    if 'Time' not in df.columns:
        raise ValueError(f"No 'Time' column in {file_path}")
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

    first_xyz_idx = next(i for i,c in enumerate(df.columns) if ' X' in c or c.endswith('X'))
    info = df.columns[:first_xyz_idx].tolist()

    cols = []
    new = {}
    for col in info:
        cols.append((col,'')); new[(col,'')] = df[col].values

    for m in marker_names:
        for axis in ['X','Y','Z']:
            cand = [c for c in df.columns if c.replace(' ','').startswith(m) and c.strip().endswith(axis)]
            if cand:
                cols.append((m,axis))
                new[(m,axis)] = df[cand[0]].values / scale_mm_to_m

    out = pd.DataFrame(new)
    out.columns = pd.MultiIndex.from_tuples(cols)
    return out

# -----------------------------
# Preprocessing
# -----------------------------
def handle_missing_linear(df: pd.DataFrame) -> pd.DataFrame:
    t = df[('Time','')].values
    for m,a in df.columns:
        if m in ('Time','Frame','SMPTE') or a=='':
            continue
        col = df[(m,a)].values.astype(float)
        bad = ~np.isfinite(col)
        if bad.any():
            good = ~bad
            if good.sum() >= 2:
                f = interp1d(t[good], col[good], kind='linear', fill_value='extrapolate', bounds_error=False)
                col[bad] = f(t[bad])
            else:
                col[bad] = np.nanmean(col)
        df[(m,a)] = col
    return df

def pelvis_center(df: pd.DataFrame) -> pd.DataFrame:
    have = [m for m in PELVIS if (m,'X') in df.columns]
    if len(have) < 2: 
        return df
    cx = df[[(m,'X') for m in have]].median(axis=1)
    cy = df[[(m,'Y') for m in have]].median(axis=1)
    cz = df[[(m,'Z') for m in have]].median(axis=1)
    for m in MARKERS:
        if (m,'X') in df.columns:
            df[(m,'X')] = df[(m,'X')] - cx
            df[(m,'Y')] = df[(m,'Y')] - cy
            df[(m,'Z')] = df[(m,'Z')] - cz
    return df

def orientation_normalize(df: pd.DataFrame) -> pd.DataFrame:
    if (NECK,'X') not in df.columns: 
        return df
    if (HIPS_PAIR[0],'X') not in df.columns or (HIPS_PAIR[1],'X') not in df.columns:
        return df

    P = np.stack([df[(NECK,a)].values for a in 'XYZ'], axis=1)
    L = np.stack([df[(HIPS_PAIR[1],a)].values for a in 'XYZ'], axis=1) 
    R = np.stack([df[(HIPS_PAIR[0],a)].values for a in 'XYZ'], axis=1) 
    lr = (L - R); lr = lr / (np.linalg.norm(lr, axis=1, keepdims=True) + EPS)

    fwd = P; fwd = fwd / (np.linalg.norm(fwd, axis=1, keepdims=True) + EPS)

    up = np.cross(lr, fwd); up = up / (np.linalg.norm(up, axis=1, keepdims=True) + EPS)
    fwd = np.cross(up, lr); fwd = fwd / (np.linalg.norm(fwd, axis=1, keepdims=True) + EPS)

    for m in MARKERS:
        if (m,'X') in df.columns:
            V = np.stack([df[(m,'X')].values, df[(m,'Y')].values, df[(m,'Z')].values], axis=1)
            Vp = np.empty_like(V)
            Vp[:,0] = np.sum(V * lr, axis=1)
            Vp[:,1] = np.sum(V * up, axis=1)
            Vp[:,2] = np.sum(V * fwd, axis=1)
            df[(m,'X')] = Vp[:,0]; df[(m,'Y')] = Vp[:,1]; df[(m,'Z')] = Vp[:,2]
    return df

def scale_normalize(df: pd.DataFrame) -> pd.DataFrame:
    if ( 'RSHD','X') in df.columns and ('LSHD','X') in df.columns:
        A = np.stack([df[('RSHD','X')],df[('RSHD','Y')],df[('RSHD','Z')]],axis=1)
        B = np.stack([df[('LSHD','X')],df[('LSHD','Y')],df[('LSHD','Z')]],axis=1)
    elif (HIPS_PAIR[0],'X') in df.columns and (HIPS_PAIR[1],'X') in df.columns:
        A = np.stack([df[(HIPS_PAIR[0],'X')],df[(HIPS_PAIR[0],'Y')],df[(HIPS_PAIR[0],'Z')]],axis=1)
        B = np.stack([df[(HIPS_PAIR[1],'X')],df[(HIPS_PAIR[1],'Y')],df[(HIPS_PAIR[1],'Z')]],axis=1)
    else:
        return df
    dist = np.linalg.norm(A - B, axis=1)
    s = np.median(dist) + EPS
    for m in MARKERS:
        if (m,'X') in df.columns:
            for a in 'XYZ':
                df[(m,a)] = df[(m,a)] / s
    return df

def resample_to(df: pd.DataFrame, target_hz: float) -> pd.DataFrame:
    t = df[('Time','')].values
    if not np.all(np.diff(t) > 0):
        t = np.maximum.accumulate(t + np.linspace(0, 1e-7, len(t)))
    t_new = np.arange(t[0], t[-1], 1.0/target_hz)
    out = {('Time',''): t_new}
    for m,a in df.columns:
        if a == '' or m == 'Time': 
            continue
        f = interp1d(t, df[(m,a)].values, kind='linear', fill_value='extrapolate', bounds_error=False)
        out[(m,a)] = f(t_new)
    new_df = pd.DataFrame(out)
    new_df.columns = pd.MultiIndex.from_tuples(list(out.keys()))
    return new_df

def smooth(df: pd.DataFrame, hz: float) -> pd.DataFrame:
    win = int(max(3, round(SMOOTH_WIN_SECS * hz)))
    if win % 2 == 0: win += 1
    poly = min(SMOOTH_POLY, win-1)
    for m,a in df.columns:
        if a in 'XYZ':
            df[(m,a)] = savgol_filter(df[(m,a)].values, window_length=win, polyorder=poly, mode='interp')
    return df

def vec(df, m): 
    return np.stack([df[(m,'X')].values, df[(m,'Y')].values, df[(m,'Z')].values], axis=1)

def joint_angle(A, B, C):
    v1 = A - B; v2 = C - B
    n1 = np.linalg.norm(v1, axis=1, keepdims=True) + EPS
    n2 = np.linalg.norm(v2, axis=1, keepdims=True) + EPS
    cos = np.sum(v1*v2, axis=1, keepdims=True) / (n1*n2)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))[:,0]

def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, slice]]:
    feats = []
    idx_map = {}

    used_markers = [m for m in MARKERS if (m,'X') in df.columns]
    start = 0
    P = []
    for m in used_markers:
        P.append(vec(df, m))
    P = np.concatenate(P, axis=1) if P else np.zeros((len(df),0))
    feats.append(P)
    end = start + P.shape[1]; idx_map['pos_all'] = slice(start, end)

    V = np.diff(P, axis=0, prepend=P[:1])
    start = end; end = start + V.shape[1]
    feats.append(V); idx_map['vel_all'] = slice(start, end)

    A = []
    for center, prox, dist in ANGLE_DEFS:
        if all((m,'X') in df.columns for m in [center, prox, dist]):
            ang = joint_angle(vec(df, prox), vec(df, center), vec(df, dist))
            A.append(ang[:,None])
    if A:
        A = np.concatenate(A, axis=1)
    else:
        A = np.zeros((len(df),0))
    start = end; end = start + A.shape[1]
    feats.append(A); idx_map['angles'] = slice(start, end)

    X = np.concatenate(feats, axis=1) if feats else np.zeros((len(df),0))

    limb_cols = {}
    for limb, markers in LIMB_GROUPS.items():
        cols = []
        for m in markers:
            if m in used_markers:
                mi = used_markers.index(m)
                cols.extend([mi*3, mi*3+1, mi*3+2])
        cols = np.array(cols)
        if len(cols) > 0:
            limb_cols[limb] = cols
    return X, idx_map | {'limb_cols': limb_cols, 'used_markers': used_markers}

def dtw_distance(A: np.ndarray, B: np.ndarray, radius=DTW_RADIUS) -> float:
    dist, _ = fastdtw(A, B, radius=radius, dist=euclidean)
    norm = max(len(A), len(B))
    return dist / (norm + EPS)

def learn_expert_model(expert_dfs: List[pd.DataFrame]) -> Dict:
    proc = []
    for df in expert_dfs:
        df = handle_missing_linear(df)
        df = pelvis_center(df)
        df = orientation_normalize(df)
        df = scale_normalize(df)
        df = resample_to(df, TARGET_HZ)
        df = smooth(df, TARGET_HZ)
        proc.append(df)

    lengths = [len(df) for df in proc]
    Tm = int(np.median(lengths))

    def retime(df):
        t_old = df[('Time','')].values
        t_new = np.linspace(t_old[0], t_old[-1], Tm)
        out = {('Time',''): t_new}
        for m,a in df.columns:
            if a in 'XYZ':
                f = interp1d(t_old, df[(m,a)].values, kind='linear', fill_value='extrapolate', bounds_error=False)
                out[(m,a)] = f(t_new)
        ndf = pd.DataFrame(out); ndf.columns = pd.MultiIndex.from_tuples(list(out.keys()))
        return ndf

    proc = [retime(df) for df in proc]

    Xs = []
    meta = None
    for df in proc:
        X, meta = build_features(df)
        Xs.append(X)

    X_ref = np.mean(np.stack(Xs, axis=0), axis=0)  

    dists = [dtw_distance(X, X_ref) for X in Xs]
    D50 = float(np.median(dists))
    D85 = float(np.percentile(dists, 85))

    return {
        'X_ref': X_ref,
        'meta': meta,
        'Tm': Tm,
        'D50': D50,
        'D85': D85,
        'target_hz': TARGET_HZ
    }

def score_player(player_df: pd.DataFrame, model: Dict) -> Dict:
    df = handle_missing_linear(player_df.copy())
    df = pelvis_center(df)
    df = orientation_normalize(df)
    df = scale_normalize(df)
    df = resample_to(df, model['target_hz'])
    df = smooth(df, model['target_hz'])

    Xp, meta = build_features(df)
    t_old = df[('Time','')].values
    Xp_ret = np.empty((model['Tm'], Xp.shape[1]))
    for k in range(Xp.shape[1]):
        f = interp1d(np.arange(len(Xp)), Xp[:,k], kind='linear', fill_value='extrapolate', bounds_error=False)
        Xp_ret[:,k] = f(np.linspace(0, len(Xp)-1, model['Tm']))

    d = dtw_distance(model['X_ref'], Xp_ret)

    D50, D85 = model['D50'] + EPS, model['D85'] + EPS

    if d <= D50:
        pct = 95.0   
    elif d >= D85:
        pct = 70.0   
    else:
        pct = 95.0 - (d - D50) / (D85 - D50) * 25.0

    pct = max(0.0, min(100.0, pct))

    limb_scores = {}
    for limb, cols in model['meta']['limb_cols'].items():
        pos_slice = model['meta']['pos_all']
        limb_cols = pos_slice.start + cols  
        d_limb = dtw_distance(model['X_ref'][:, limb_cols], Xp_ret[:, limb_cols])

        if d_limb <= D50:
            limb_pct = 95.0
        elif d_limb >= D85:
            limb_pct = 70.0
        else:
            limb_pct = 95.0 - (d_limb - D50) / (D85 - D50) * 25.0

        limb_scores[limb] = float(max(0.0, min(100.0, limb_pct)))

    NSEG = 10
    seg_scores = []
    T = model['Tm']
    Xr = model['X_ref']
    for i in range(NSEG):
        a = int(round(i*T/NSEG)); b = int(round((i+1)*T/NSEG)); 
        if b <= a: b = a+1
        d_seg = dtw_distance(Xr[a:b], Xp_ret[a:b])

        if d_seg <= D50:
            seg_pct = 95.0
        elif d_seg >= D85:
            seg_pct = 70.0
        else:
            seg_pct = 95.0 - (d_seg - D50) / (D85 - D50) * 25.0

        seg_scores.append(float(max(0.0, min(100.0, seg_pct))))

    tips = []
    worst_limb = min(limb_scores, key=limb_scores.get)
    if limb_scores[worst_limb] < 80:
        tips.append(f"Focus on {worst_limb.lower()} tracking and range; mirror the reference slowly, then speed up.")
    worst_seg = int(np.argmin(seg_scores))
    tips.append(f"Your weakest segment is {worst_seg+1}/{NSEG}. Rewatch that portion and rehearse it 2–3 times.")

    return {
        'overall_percent': round(pct, 2),
        'dtw_distance': float(d),
        'limb_scores': limb_scores,
        'segment_scores': seg_scores,
        'tips': tips
    }
