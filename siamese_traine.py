# siamese_trainer.py
import os
import random
import json
import csv
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, random_split


# ------------------------------------------------------------
# Repro & small utilities
# ------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

set_seed(42)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
TARGET_LEN = 600   # fixed frames (20s @30Hz)
MARKERS = [
    'BKHD','RFHD','LFHD','C7','NCK','RSHD','LSHD','RELB','LELB','RWRS','LWRS',
    'RIND','LIND','RPNK','LPNK','RFHP','RBHP','LFHP','LBHP','RKNE','LKNE',
    'RFAK','RBAK','LFAK','LBAK'
]
LIMB_GROUPS = {
    "Right Arm": ['RSHD','RELB','RWRS','RIND','RPNK'],
    "Left Arm":  ['LSHD','LELB','LWRS','LIND','LPNK'],
    "Right Leg": ['RFHP','RKNE','RFAK','RBAK'],
    "Left Leg":  ['LFHP','LKNE','LFAK','LBAK'],
    "Torso/Head":['BKHD','RFHD','LFHD','C7','NCK','RBHP','LBHP']
}

# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------
def load_qualisys_tsv(file_path: str, scale_factor=1000.0) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Frame"):
            header_index = i
            break

    if header_index is None:
        header_index = 0  

    df = pd.read_csv(file_path, sep="\t", header=0, skiprows=header_index)

    if "Time" not in df.columns:
        raise ValueError(f"No Time column found in {file_path}")

    cols = []
    for m in MARKERS:
        for c in ["X", "Y", "Z"]:
            name1 = f"{m}_{c}"   
            name2 = f"{m} {c}"   
            if name1 in df.columns:
                cols.append(name1)
            elif name2 in df.columns:
                cols.append(name2)

    return df[["Time"] + cols]


def preprocess_and_fix_length(file_path: str, target_len=TARGET_LEN) -> Tuple[np.ndarray, np.ndarray]:
    df = load_qualisys_tsv(file_path)

    mean_abs = np.abs(df.select_dtypes(float)).mean().mean()
    if mean_abs < 10:      
        scale_factor = 1.0
    else:                  
        scale_factor = 1.0 / 1000.0

    T = len(df)
    data = []

    for m in MARKERS:
        cols = [f"{m} X", f"{m} Y", f"{m} Z"]
        if all(c in df.columns for c in cols):
            xyz = df[cols].values * scale_factor
        else:
            xyz = np.zeros((T, 3))
        data.append(xyz)

    X = np.stack(data, axis=1)  # (T, J, 3)

    pelvis_idx = [MARKERS.index(x) for x in ["RBHP", "LBHP"] if x in MARKERS]
    if pelvis_idx:
        pelvis_center = X[:, pelvis_idx, :3].mean(axis=1, keepdims=True)
        X[:, :, :3] -= pelvis_center

    vel = np.gradient(X, axis=0)
    X = np.concatenate([X, vel], axis=2)  # (T, J, 6)

    energy = np.sqrt((vel ** 2).sum(axis=2)).mean(axis=1)
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    mask_active = energy > 0.02        
    if mask_active.any():
        s, e = np.where(mask_active)[0][[0, -1]]
        s = max(0, s - int(0.05 * T))  
        e = min(T - 1, e + int(0.05 * T))
        X = X[s:e+1]
    T = len(X)

    t_old = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, target_len)
    Xn = np.zeros((target_len, X.shape[1], X.shape[2]))
    for j in range(X.shape[1]):
        for c in range(X.shape[2]):
            Xn[:, j, c] = np.interp(t_new, t_old, X[:, j, c])

    mask = np.ones((target_len,), dtype=np.float32)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)

    mean = Xn.mean(axis=(0, 1), keepdims=True)
    std  = Xn.std(axis=(0, 1), keepdims=True) + 1e-6
    Xn = (Xn - mean) / std

    return Xn.astype(np.float32), mask


# ------------------------------------------------------------
# Augmentations
# ------------------------------------------------------------
def aug_time_warp(X: np.ndarray, max_scale: float = 0.08) -> np.ndarray:
    T = X.shape[0]
    curve = np.cumsum(np.random.randn(T))
    curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-8)
    scale = 1.0 + (curve - 0.5) * (2 * max_scale)
    t_new = np.cumsum(scale)
    t_new = (t_new - t_new.min()) / (t_new.max() - t_new.min() + 1e-8)
    grid = np.linspace(0, 1, T)
    Xw = np.empty_like(X)
    for j in range(X.shape[1]):
        for c in range(X.shape[2]):
            Xw[:, j, c] = np.interp(grid, t_new, X[:, j, c])
    return Xw

def aug_global_rotate(X: np.ndarray, max_deg=5.0) -> np.ndarray:
    theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    axis = np.random.randn(3); axis /= (np.linalg.norm(axis) + 1e-8)
    ux, uy, uz = axis
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c+ux*ux*(1-c),     ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
        [uy*ux*(1-c)+uz*s,  c+uy*uy*(1-c),    uy*uz*(1-c)-ux*s],
        [uz*ux*(1-c)-uy*s,  uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)]
    ], dtype=np.float32)
    Xr = X.copy()
    P = X[:, :, 0:3].reshape(-1, 3) @ R.T
    Xr[:, :, 0:3] = P.reshape(X.shape[0], X.shape[1], 3)
    return Xr

def aug_scale(X: np.ndarray, scale_range=(0.95, 1.05)) -> np.ndarray:
    s = np.random.uniform(*scale_range)
    Xs = X.copy(); Xs[:, :, 0:3] *= s
    return Xs

def aug_mirror(X: np.ndarray) -> np.ndarray:
    Xm = X.copy(); Xm[:, :, 0] *= -1  
    return Xm

def aug_add_noise(X: np.ndarray, pos_sigma=0.003, vel_sigma=0.008) -> np.ndarray:
    Xn = X.copy()
    Xn[:, :, 0:3] += np.random.randn(*X[:, :, 0:3].shape) * pos_sigma
    if X.shape[2] > 3:
        Xn[:, :, 3:6] += np.random.randn(*X[:, :, 3:6].shape) * vel_sigma
    return Xn

def aug_drop_jitter(X: np.ndarray, drop_prob=0.01) -> np.ndarray:
    Xd = X.copy()
    mask = (np.random.rand(*X[:, :, 0:1].shape) < drop_prob)
    Xd[:, :, 0:3][mask.repeat(3, axis=2)] = 0.0
    return Xd

def aug_random_trim(X, keep_ratio=(0.7, 1.0), target_len=TARGET_LEN):
    T = X.shape[0]
    k = int(T * np.random.uniform(*keep_ratio))
    if k >= T:
        return X
    s = np.random.randint(0, T - k)
    X_trim = X[s:s+k]

    t_old = np.linspace(0, 1, len(X_trim))
    t_new = np.linspace(0, 1, target_len)
    X_resamp = np.zeros((target_len, X.shape[1], X.shape[2]))
    for j in range(X.shape[1]):
        for c in range(X.shape[2]):
            X_resamp[:, j, c] = np.interp(t_new, t_old, X_trim[:, j, c])
    return X_resamp

def augment_sequence(X: np.ndarray, target_len=TARGET_LEN) -> np.ndarray:
    X2 = X
    if np.random.rand() < 0.3:
        X2 = aug_random_trim(X2, target_len=target_len)
    if np.random.rand() < 0.5:
        X2 = aug_time_warp(X2, 0.08)
    if np.random.rand() < 0.5:
        X2 = aug_global_rotate(X2, 5.0)
    if np.random.rand() < 0.5:
        X2 = aug_scale(X2, (0.95, 1.05))
    if np.random.rand() < 0.2:
        X2 = aug_mirror(X2)
    if np.random.rand() < 0.7:
        X2 = aug_add_noise(X2, 0.003, 0.008)
    if np.random.rand() < 0.2:
        X2 = aug_drop_jitter(X2, 0.01)
    return X2


# ------------------------------------------------------------
# Datasets & Models
# ------------------------------------------------------------
class SiamesePairsDataset(Dataset):
    def __init__(self, hy_files, bd_files, target_len=TARGET_LEN, cache=True):
        self.hy_files = hy_files
        self.bd_files = bd_files
        self.target_len = target_len
        self.cache = cache
        if cache:
            self.hy_data = [preprocess_and_fix_length(p, target_len) for p in self.hy_files]
            self.bd_data = [preprocess_and_fix_length(p, target_len) for p in self.bd_files]
        self.length = max(len(self.hy_files), len(self.bd_files)) * 2

    def __len__(self): 
        return self.length
    
    def __getitem__(self, idx):
        alike = random.random() < 0.5
        if alike:
            if self.cache:
                X1, m1 = random.choice(self.hy_data); X2, m2 = random.choice(self.hy_data)
            else:
                X1, m1 = preprocess_and_fix_length(random.choice(self.hy_files), self.target_len)
                X2, m2 = preprocess_and_fix_length(random.choice(self.hy_files), self.target_len)
            label = 1
            if random.random() < 0.5:
                X1 = augment_sequence(X1)
            else:
                X2 = augment_sequence(X2)
        else:
            if self.cache:
                X1, m1 = random.choice(self.hy_data); X2, m2 = random.choice(self.bd_data)
            else:
                X1, m1 = preprocess_and_fix_length(random.choice(self.hy_files), self.target_len)
                X2, m2 = preprocess_and_fix_length(random.choice(self.bd_files), self.target_len)
            label = 0
            X1 = augment_sequence(X1)

        return (
            torch.from_numpy(X1).float(),
            torch.from_numpy(m1).float(),
            torch.from_numpy(X2).float(),
            torch.from_numpy(m2).float(),
            torch.tensor(label, dtype=torch.float32),
        )

class StratifiedPairsDataset(Dataset):
    def __init__(self, hy_files, bd_files, target_len=TARGET_LEN, cache=True):
        self.hy_files = hy_files
        self.bd_files = bd_files
        self.target_len = target_len
        self.cache = cache
        if cache:
            self.hy_data = [preprocess_and_fix_length(p, target_len) for p in self.hy_files]
            self.bd_data = [preprocess_and_fix_length(p, target_len) for p in self.bd_files]
        self.pairs = []
        for i in range(len(self.hy_files)):
            for j in range(i+1, len(self.hy_files)):
                self.pairs.append(("hy", i, j, 1))
        for i in range(len(self.hy_files)):
            for j in range(len(self.bd_files)):
                self.pairs.append(("mix", i, j, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_type, i, j, label = self.pairs[idx]
        if pair_type == "hy":
            if self.cache:
                X1, m1 = self.hy_data[i]; X2, m2 = self.hy_data[j]
            else:
                X1, m1 = preprocess_and_fix_length(self.hy_files[i], self.target_len)
                X2, m2 = preprocess_and_fix_length(self.hy_files[j], self.target_len)
        else:
            if self.cache:
                X1, m1 = self.hy_data[i]; X2, m2 = self.bd_data[j]
            else:
                X1, m1 = preprocess_and_fix_length(self.hy_files[i], self.target_len)
                X2, m2 = preprocess_and_fix_length(self.bd_files[j], self.target_len)
        return (
            torch.from_numpy(X1).float(),
            torch.from_numpy(m1).float(),
            torch.from_numpy(X2).float(),
            torch.from_numpy(m2).float(),
            torch.tensor(label, dtype=torch.float32),
        )

class MotionTransformer(nn.Module):
    def __init__(self, J=25, C=6, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(J*C, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        B,T,J,C = x.shape
        x = x.reshape(B,T,J*C)
        x = self.proj(x)
        pad_mask = (mask==0.0) if mask is not None else None
        z = self.encoder(x, src_key_padding_mask=pad_mask)
        z = self.norm(z)
        if mask is not None:
            denom = mask.sum(dim=1,keepdim=True).clamp(min=1.0)
            z = (z*mask.unsqueeze(-1)).sum(dim=1)/denom
        else:
            z = z.mean(dim=1)
        return F.normalize(z, p=2, dim=1, eps=1e-6)

class SiameseNet(nn.Module):
    def __init__(self, J=25, C=6, d_model=128):
        super().__init__()
        self.encoder = MotionTransformer(J, C, d_model)
        self.reg_head = nn.Sequential(nn.Linear(d_model*3,128), nn.ReLU(), nn.Linear(128,1))
    def forward(self, X1, m1, X2, m2):
        e1 = self.encoder(X1, m1); e2 = self.encoder(X2, m2)
        feat = torch.cat([torch.abs(e1-e2), e1*e2, e1+e2], dim=1)
        sim = torch.sigmoid(self.reg_head(feat)).clamp(1e-6, 1-1e-6) * 100.0
        return e1, e2, sim.squeeze(1)

def contrastive_loss(e1, e2, y, margin=1.0):
    dist = F.pairwise_distance(e1, e2)
    pos = y * dist.pow(2)
    neg = (1 - y) * F.relu(margin - dist).pow(2)
    return (pos + neg).mean(), dist

def regression_loss(sim, target):
    return F.l1_loss(sim, target)

def triplet_loss(e1, e2, en, margin=1.0):
    pos_dist = F.pairwise_distance(e1, e2)
    neg_dist = F.pairwise_distance(e1, en)
    losses = F.relu(pos_dist - neg_dist + margin)
    return losses.mean()
