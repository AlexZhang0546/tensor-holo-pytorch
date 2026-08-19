#!/bin/bash
# Single-sample stage2 quick verification (RealAmpPhaseDDPMNet).
set -e
cd /root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch
/root/autodl-tmp/miniconda3/envs/holography/bin/python src/_ddpm_overfit.py 300 300 1e-4 2>&1 | tee ddpm_overfit.log
