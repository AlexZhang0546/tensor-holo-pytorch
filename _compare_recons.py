import os
import sys

import cv2
import numpy as np


def ssim(a, b, data_range=255.0):
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim == 3:
        return float(np.mean([
            ssim(a[:, :, c], b[:, :, c], data_range=data_range)
            for c in range(a.shape[2])
        ]))
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    mu1 = cv2.filter2D(a, -1, window, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.filter2D(b, -1, window, borderType=cv2.BORDER_REFLECT)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = cv2.filter2D(a * a, -1, window, borderType=cv2.BORDER_REFLECT) - mu1_sq
    sigma2_sq = cv2.filter2D(b * b, -1, window, borderType=cv2.BORDER_REFLECT) - mu2_sq
    sigma12 = cv2.filter2D(a * b, -1, window, borderType=cv2.BORDER_REFLECT) - mu12
    return float(np.mean(
        ((2 * mu12 + c1) * (2 * sigma12 + c2)) /
        ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ))


def psnr(a, b, data_range=255.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def grayscale_rgb(img_bgr):
    # RGB luminance using Rec. 601, but keep it simple and channel-order agnostic.
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def compare_one(name, recon_path, target_path):
    recon = cv2.imread(recon_path, cv2.IMREAD_COLOR)
    target = cv2.imread(target_path, cv2.IMREAD_COLOR)
    if recon.shape[:2] != target.shape[:2]:
        target = cv2.resize(target, (recon.shape[1], recon.shape[0]),
                            interpolation=cv2.INTER_CUBIC)
    out = {
        "name": name,
        "ssim_rgb": ssim(recon, target),
        "psnr_rgb": psnr(recon, target),
        "ssim_gray": ssim(grayscale_rgb(recon), grayscale_rgb(target)),
        "psnr_gray": psnr(grayscale_rgb(recon), grayscale_rgb(target)),
        "channel_ssim": [ssim(recon[:, :, i], target[:, :, i])
                         for i in range(3)],
        "channel_psnr": [psnr(recon[:, :, i], target[:, :, i])
                         for i in range(3)],
    }
    return out


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    original_dir = r"C:\Users\zrx07\Documents\program\tensor-holo"
    target = os.path.join(original_dir, "data", "example_input", "bbb_rgb.png")
    rows = []
    rows.append(compare_one(
        "original_amp_filtered",
        os.path.join(original_dir, "output_original_bbb", "amp_filtered.png"),
        target))
    orig_recon = cv2.imread(
        os.path.join(original_dir, "output_original_bbb", "amp_filtered.png"),
        cv2.IMREAD_COLOR)
    target_for_orig = cv2.imread(target, cv2.IMREAD_COLOR)
    target_for_orig = cv2.resize(
        target_for_orig, (orig_recon.shape[1], orig_recon.shape[0]),
        interpolation=cv2.INTER_CUBIC)
    rows.append({
        "name": "original_amp_filtered_flip_v",
        "ssim_rgb": ssim(orig_recon[::-1], target_for_orig),
        "psnr_rgb": psnr(orig_recon[::-1], target_for_orig),
        "ssim_gray": ssim(grayscale_rgb(orig_recon[::-1]),
                          grayscale_rgb(target_for_orig)),
        "psnr_gray": psnr(grayscale_rgb(orig_recon[::-1]),
                          grayscale_rgb(target_for_orig)),
        "channel_ssim": [ssim(orig_recon[::-1][:, :, i], target_for_orig[:, :, i])
                         for i in range(3)],
        "channel_psnr": [psnr(orig_recon[::-1][:, :, i], target_for_orig[:, :, i])
                         for i in range(3)],
    })
    rows.append({
        "name": "original_amp_filtered_flip_v_swap",
        "ssim_rgb": ssim(orig_recon[::-1][:, :, ::-1], target_for_orig),
        "psnr_rgb": psnr(orig_recon[::-1][:, :, ::-1], target_for_orig),
        "ssim_gray": ssim(grayscale_rgb(orig_recon[::-1][:, :, ::-1]),
                          grayscale_rgb(target_for_orig)),
        "psnr_gray": psnr(grayscale_rgb(orig_recon[::-1][:, :, ::-1]),
                          grayscale_rgb(target_for_orig)),
        "channel_ssim": [ssim(orig_recon[::-1][:, :, ::-1][:, :, i],
                              target_for_orig[:, :, i])
                         for i in range(3)],
        "channel_psnr": [psnr(orig_recon[::-1][:, :, ::-1][:, :, i],
                              target_for_orig[:, :, i])
                         for i in range(3)],
    })
    rows.append(compare_one(
        "improved_recon_rgb",
        os.path.join(base, "output_bbb", "recon_rgb.png"),
        target))
    # Also compare with swapped channels, in case the improved writer reversed BGR/RGB.
    recon = cv2.imread(os.path.join(base, "output_bbb", "recon_rgb.png"), cv2.IMREAD_COLOR)
    target_img = cv2.imread(target, cv2.IMREAD_COLOR)
    target_img = cv2.resize(target_img, (recon.shape[1], recon.shape[0]),
                            interpolation=cv2.INTER_CUBIC)
    rows.append({
        "name": "improved_recon_rgb_swapped",
        "ssim_rgb": ssim(recon[:, :, ::-1], target_img),
        "psnr_rgb": psnr(recon[:, :, ::-1], target_img),
        "ssim_gray": ssim(grayscale_rgb(recon[:, :, ::-1]), grayscale_rgb(target_img)),
        "psnr_gray": psnr(grayscale_rgb(recon[:, :, ::-1]), grayscale_rgb(target_img)),
        "channel_ssim": [ssim(recon[:, :, ::-1][:, :, i], target_img[:, :, i])
                         for i in range(3)],
        "channel_psnr": [psnr(recon[:, :, ::-1][:, :, i], target_img[:, :, i])
                         for i in range(3)],
    })
    for r in rows:
        print(f"{r['name']}:")
        print(f"  SSIM_rgb={r['ssim_rgb']:.4f}  PSNR_rgb={r['psnr_rgb']:.2f} dB")
        print(f"  SSIM_gray={r['ssim_gray']:.4f}  PSNR_gray={r['psnr_gray']:.2f} dB")
        print(f"  channel_ssim={[f'{v:.4f}' for v in r['channel_ssim']]}")
        print(f"  channel_psnr={[f'{v:.2f}' for v in r['channel_psnr']]}")


if __name__ == "__main__":
    main()
