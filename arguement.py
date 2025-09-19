import numpy as np
from typing import Tuple, List, Dict, Any
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

# ========== 工具函数 ==========
def _pick_anom_span(T: int, window_size: int) -> Tuple[int, int]:
    """在长度 T 上随机挑一个连续区间，长度 ~ [T//20, T//3)。"""
    lo = max(1, window_size // 5)
    hi = max(lo + 1, window_size // 3)
    anom_len = int(np.random.randint(lo, hi))
    if anom_len >= T:
        return 0, T
    start = int(np.random.randint(0, T - anom_len + 1))
    return start, start + anom_len

def _safe_minmax_like(original_1d: np.ndarray, arr_1d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    将 arr_1d 线性映射到 original_1d 的全局 [min, max]。
    - 若 original 为常数（max-min < eps），则直接返回一个常数向量（该常数=original 的值）。
    - 若 arr 为常数，也返回 original 区间中点（避免除零）。
    - 自动处理 NaN/Inf。
    """
    o = np.asarray(original_1d, dtype=float).ravel()
    a = np.asarray(arr_1d, dtype=float).ravel()

    # 替换 NaN/Inf，防止传播
    o = np.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    o_min, o_max = float(np.min(o)), float(np.max(o))
    o_rng = o_max - o_min

    if o_rng < eps:
        # 原序列本身是常数 → 返回同长度常数向量
        return np.full_like(a, fill_value=o_min, dtype=float)

    a_min, a_max = float(np.min(a)), float(np.max(a))
    a_rng = a_max - a_min

    if a_rng < eps:
        # 目标序列也几乎是常数 → 映射到原区间的中点
        mid = (o_min + o_max) / 2.0
        return np.full_like(a, fill_value=mid, dtype=float)

    # 标准 min-max 映射
    a_norm = (a - a_min) / a_rng
    out = a_norm * o_rng + o_min
    return out

# ========== 四种单变量增强 ==========
def aug_jittering(ts: np.ndarray, window_size: int, noise_scale: float = 0.2):
    """
    在随机区间内，用 (ts + N(0, sigma)) 替换，其中 sigma = (max-min)*noise_scale
    """
    ts = np.asarray(ts, dtype=float)
    T = ts.size
    s, e = _pick_anom_span(T, window_size)
    sigma = (ts.max() - ts.min()) * float(noise_scale)
    noisy = ts + sigma * np.random.randn(T)
    out = ts.copy()
    out[s:e] = noisy[s:e]
    return out, (s, e)

def aug_warping(ts: np.ndarray, window_size: int, fs: float = 1.0):
    """
    低频带通滤波 + MinMax 回标，再随机区间替换。
    说明：通过 FFT/PSD 选低频候选，设计 Butter 带通（显式传 fs，避免归一化混淆）
    """
    ts = np.asarray(ts, dtype=float)
    T = ts.size
    s, e = _pick_anom_span(T, window_size)

    # FFT & PSD
    fft_vals = np.fft.fft(ts)
    psd = np.abs(fft_vals) ** 2
    topk = min(30, T)
    peak_idx = np.argsort(psd)[-topk:]
    freqs = np.fft.fftfreq(T, d=1.0 / fs)  # Hz
    cand = np.unique(np.sort(freqs[peak_idx]))
    cand = cand[cand > 0]  # 只用正频

    nyq = fs / 2.0
    eps = 1e-3
    if cand.size >= 2:
        pick_idx = np.arange(0, cand.size, 3)
        pool = cand[pick_idx][:4] if cand.size >= 4 else cand[pick_idx]
        if pool.size < 2:
            pool = cand[: min(4, cand.size)]
        low_hz, high_hz = np.random.choice(pool, size=2, replace=False)
        low_hz, high_hz = float(min(low_hz, high_hz)), float(max(low_hz, high_hz))
        low_hz = max(eps, min(low_hz, nyq - 2 * eps))
        high_hz = max(low_hz + eps, min(high_hz, nyq - eps))
    else:
        # 兜底：很低的带宽
        low_hz, high_hz = 0.02 * fs, 0.08 * fs
        low_hz = max(eps, min(low_hz, nyq - 2 * eps))
        high_hz = max(low_hz + eps, min(high_hz, nyq - eps))

    b, a = signal.butter(4, [low_hz, high_hz], btype='band', fs=fs)
    filt = signal.lfilter(b, a, ts)
    filt = _safe_minmax_like(ts, filt)

    out = ts.copy()
    out[s:e] = filt[s:e]
    return out, (s, e)

def aug_scaling(ts: np.ndarray, window_size: int, low: float = 0.5, high: float = 1.5, smooth: bool = False):
    """
    在随机区间内用缩放后的信号替换。
    smooth=True 会对缩放因子做一次低通，使之更像缓慢幅值漂移。
    """
    ts = np.asarray(ts, dtype=float)
    T = ts.size
    s, e = _pick_anom_span(T, window_size)

    factors = np.random.uniform(low, high, T)
    if smooth:
        b, a = signal.butter(2, 0.1, btype='lowpass')  # 0.1 of Nyquist
        factors = signal.filtfilt(b, a, factors)

    scaled = ts * factors
    out = ts.copy()
    out[s:e] = scaled[s:e]
    return out, (s, e)

def aug_permutation(ts: np.ndarray, window_size: int, min_seg: int = 3, max_seg: int = 9):
    """
    将序列前缀分成等长段打乱，余数保留在末尾，再随机区间替换。
    """
    ts = np.asarray(ts, dtype=float)
    T = ts.size
    s, e = _pick_anom_span(T, window_size)

    max_seg = min(max_seg, T)
    num_segments = int(np.random.randint(min_seg, max_seg + 1))
    seg_len = max(1, T // num_segments)
    usable = seg_len * num_segments

    segments = [ts[i:i + seg_len] for i in range(0, usable, seg_len)]
    np.random.shuffle(segments)
    prefix = np.concatenate(segments) if segments else np.array([], dtype=float)
    remainder = ts[usable:]
    permuted = np.concatenate([prefix, remainder])

    out = ts.copy()
    out[s:e] = permuted[s:e]
    return out, (s, e)

# 在文件顶部补充
import numpy as np
import torch

def make_negative_multivariate(
    X,
    fs: float = 1.0,
    max_dim_ratio: float = 0.9,
    noise_scale: float = 0.7,
    scaling_range = (1.5, 2.5),
    smooth_scaling: bool = False,
    seed: int = None,
    return_mask: bool = True
):
    """
    若 X 是 torch.Tensor：返回 torch.Tensor（保持原 device & dtype）
    若 X 是 numpy.ndarray：返回 numpy.ndarray（保持原行为）
    其他逻辑与原实现一致
    """
    # 记录输入类型与可能的设备/精度
    is_torch_input = torch.is_tensor(X)
    if is_torch_input:
        in_device = X.device
        in_dtype  = X.dtype
        X_np = X.detach().cpu().numpy()
    elif isinstance(X, np.ndarray):
        X_np = X
    else:
        raise TypeError("X must be a torch.Tensor or numpy.ndarray")

    # ====== 原先的 numpy 实现从这里开始 ======
    assert X_np.ndim == 3, "X must be [B, T, D]"
    B, T, D = X_np.shape
    window_size = T

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    X_neg_np = X_np.astype(float).copy()
    mask_np = np.zeros_like(X_neg_np, dtype=np.uint8)
    info = []

    for b in range(B):
        max_k = max(D//3, int(np.floor(D * max_dim_ratio)))
        k = int(rng.integers(D//3, max_k + 1))
        chosen = rng.choice(D, size=k, replace=False).tolist()

        details = []
        for d in chosen:
            ts = X_neg_np[b, :, d]
            aug_name = rng.choice(["jittering", "warping", "scaling", "permutation"])
            if aug_name == "jittering":
                mod, (s, e) = aug_jittering(ts, window_size, noise_scale=noise_scale)
            elif aug_name == "warping":
                mod, (s, e) = aug_warping(ts, window_size, fs=fs)
            elif aug_name == "scaling":
                lo, hi = float(scaling_range[0]), float(scaling_range[1])
                mod, (s, e) = aug_scaling(ts, window_size, low=lo, high=hi, smooth=smooth_scaling)
            else:
                mod, (s, e) = aug_permutation(ts, window_size)

            X_neg_np[b, :, d] = mod
            if return_mask:
                mask_np[b, s:e, d] = 1
            details.append({"dim": int(d), "aug": str(aug_name), "span": (int(s), int(e))})

        info.append({"chosen_dims": chosen, "details": details})
    # ====== 原先的 numpy 实现到这里结束 ======

    # 恢复为与输入一致的类型
    if is_torch_input:
        X_neg = torch.from_numpy(X_neg_np).to(device=in_device, dtype=in_dtype)
        mask  = torch.from_numpy(mask_np).to(device=in_device) if return_mask else torch.zeros_like(X, dtype=torch.uint8)
        return X_neg, info, mask
    else:
        return X_neg_np, info, mask_np