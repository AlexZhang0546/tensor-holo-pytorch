# -*- coding: utf-8 -*-
"""DPM roundtrip verification: torch impl vs numpy TF-exact reference.
No depth shift; proper metrics (|corr|, amplitude corr, norm amp err).
"""
import sys, os, numpy as np, torch
sys.path.insert(0, "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch")
from src.optics.complex_utils import compl_val
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only

RES = int(sys.argv[1]) if len(sys.argv) > 1 else 192
RAD = RES // 2
torch.manual_seed(1)
np.random.seed(1)

# smooth random complex field (3, H, W)
r = np.random.randn(3, RES, RES)
s = np.random.randn(3, RES, RES)
from numpy import fft as npfft
# smooth with simple box blur in numpy
def box_blur(x, k=7):
    out = np.zeros_like(x)
    p = k // 2
    xp = np.pad(x, ((0,0),(p,p),(p,p)), mode='edge')
    for i in range(k):
        for j in range(k):
            out += xp[:, i:i+RES, j:j+RES]
    return out / (k*k)
r = box_blur(r); s = box_blur(s)
amp = np.abs(r) + 0.5
phs = s / (s.std() + 1e-9) * 1.5
rt_np = amp * np.exp(1j * phs)          # (3, H, W)

# ---------------- numpy TF-exact reference ----------------
def np_aadpm(cpx):
    amp = np.abs(cpx)
    phs = np.angle(cpx)
    amp_max = amp.max() + 1e-6
    amp = amp / amp_max
    amp = np.minimum(amp, 1.0 - 1e-6)
    phs_zero_mean = phs - phs.mean(axis=(1, 2), keepdims=True)
    phs_offset = np.arccos(amp)
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset
    H, W = phs.shape[1:]
    phs_only = np.zeros((3, H, W))
    phs_only[:, 0::2, 0::2] = phs_low[:, 0::2, 0::2]
    phs_only[:, 0::2, 1::2] = phs_high[:, 0::2, 1::2]
    phs_only[:, 1::2, 0::2] = phs_high[:, 1::2, 0::2]
    phs_only[:, 1::2, 1::2] = phs_low[:, 1::2, 1::2]
    return phs_only, amp_max

def np_filter_phs_only(phs_only, amp_max):
    cpx = amp_max * np.exp(1j * phs_only)     # (3,H,W)
    F = npfft.fft2(cpx, axes=(-2, -1))
    F = npfft.fftshift(F, axes=(-2, -1))
    y, x = np.meshgrid(np.linspace(-(RES-1)/2, (RES-1)/2, RES),
                       np.linspace(-(RES-1)/2, (RES-1)/2, RES), indexing='ij')
    mask = (x**2 + y**2) <= RAD**2
    F *= mask
    g = npfft.ifft2(npfft.ifftshift(F, axes=(-2, -1)), axes=(-2, -1))
    return g

phs_only_np, amp_max_np = np_aadpm(rt_np)
rt_enc_np = np_filter_phs_only(phs_only_np, amp_max_np)

# ---------------- torch impl ----------------
rt_t = torch.tensor(rt_np, dtype=torch.complex64).unsqueeze(0)
with torch.no_grad():
    phs_only_t, amp_max_t = aadpm(rt_t, depth_shift=0.0, sigma=0.0, kernel_width=3,
                                  phs_max=None, amp_max=None, clamp=True, normalize=False)
    amp_f, phs_f = filter_phs_only(phs_only_t, unnormalize_input=False, normalize_output=False,
                                   depth_shift=0.0, radius=RAD, res_h=RES, res_w=RES, phs_max=None, amp_max=amp_max_t)
rt_enc_t = compl_val(amp_f, phs_f).squeeze(0).numpy()

def metrics(a, b, tag):
    a = a.astype(np.complex64); b = b.astype(np.complex64)
    corr = np.abs(np.sum(np.conj(a) * b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    # optimal global scale+phase alignment
    alpha = np.sum(np.conj(a) * b) / np.sum(np.conj(a) * a)
    b_aligned = alpha * a
    amp_corr = np.corrcoef(np.abs(a).ravel(), np.abs(b).ravel())[0, 1]
    amp_err = np.mean(np.abs(np.abs(a) - np.abs(b))) / (np.mean(np.abs(a)) + 1e-9)
    phs_err = np.mean(np.abs(np.angle(np.conj(b_aligned) * b))) / np.pi
    print("[%s] |corr| %.4f amp_corr %.4f amp_rel_err %.4f phs_err(pi) %.4f  ampGT %.4f ampEnc %.4f" % (
        tag, corr, amp_corr, amp_err, phs_err, np.mean(np.abs(a)), np.mean(np.abs(b))), flush=True)

metrics(rt_np, rt_enc_np, "numpy-TF")
metrics(rt_np, rt_enc_t, "torch ")
# also compare numpy vs torch outputs directly
metrics(rt_enc_np, rt_enc_t, "np-vs-torch")
print("phs_only diff max: %.6f" % np.abs(phs_only_np - phs_only_t.squeeze(0).numpy()).max(), flush=True)
print("amp_max np %.6f torch %.6f" % (amp_max_np, amp_max_t.squeeze().numpy()[0] if amp_max_t.squeeze().dim() else amp_max_t.item()), flush=True)
print("DONE", flush=True)
