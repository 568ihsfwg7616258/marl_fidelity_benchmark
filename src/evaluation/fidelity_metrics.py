# src/evaluation/fidelity_metrics.py
import numpy as np
from scipy.stats import spearmanr

def measure_fidelity(gt, xai):
    gt = np.array(gt)
    xai = np.array(xai)
    
    # نرمال‌سازی
    gt = gt / (np.linalg.norm(gt) + 1e-8)
    xai = xai / (np.linalg.norm(xai) + 1e-8)
    
    cosine = np.dot(gt, xai)
    spearman, _ = spearmanr(gt, xai)
    
    k = 5
    gt_top = np.argsort(-gt)[:k]
    xai_top = np.argsort(-xai)[:k]
    topk = len(set(gt_top) & set(xai_top)) / k
    
    return {"cosine": cosine, "spearman": spearman if not np.isnan(spearman) else 0, "topk": topk}