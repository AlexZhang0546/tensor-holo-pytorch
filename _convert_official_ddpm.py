# -*- coding: utf-8 -*-
"""Convert the official TF ddpm_12 DDPM weights to a numpy archive."""
import tensorflow as tf
import numpy as np

CKPT = "model/ckpt_full_loss_pitch_8_layers_30_filters_24_ldi_0_ddpm_12/ckpt-7695000"
r = tf.train.NewCheckpointReader(CKPT)
names = r.get_variable_to_shape_map()

out = {}
for i in range(8):
    wname = "ddpm/Variable" if i == 0 else "ddpm/Variable_%d" % (2 * i)
    bname = "ddpm/Variable_%d" % (2 * i + 1)
    bnname = "ddpm/batch_normalization" if i == 0 else "ddpm/batch_normalization_%d" % i
    w = r.get_tensor(wname)
    b = r.get_tensor(bname)
    gamma = r.get_tensor(bnname + "/gamma")
    beta = r.get_tensor(bnname + "/beta")
    # TF [H, W, in, out] -> torch [out, in, H, W]
    out["conv%d_w" % i] = np.transpose(w, (3, 2, 0, 1)).astype(np.float32)
    out["conv%d_b" % i] = b.astype(np.float32)
    out["bn%d_gamma" % i] = gamma.astype(np.float32)
    out["bn%d_beta" % i] = beta.astype(np.float32)
    print(wname, w.shape, "->", out["conv%d_w" % i].shape)

np.savez("official_ddpm.npz", **out)
print("saved official_ddpm.npz")
