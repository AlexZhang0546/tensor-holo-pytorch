import tensorflow as tf
import numpy as np
r = tf.train.NewCheckpointReader("model/ckpt_full_loss_pitch_8_layers_30_filters_24_ldi_0_ddpm_12/ckpt-7695000")
for i in range(8):
    name = "ddpm/batch_normalization" if i == 0 else "ddpm/batch_normalization_%d" % i
    mm = r.get_tensor(name + "/moving_mean")
    mv = r.get_tensor(name + "/moving_variance")
    g = r.get_tensor(name + "/gamma")
    b = r.get_tensor(name + "/beta")
    print("BN%d" % i, "mm:", np.round(mm, 4), "mv:", np.round(mv, 4), "gamma:", np.round(g, 4), "beta:", np.round(b, 4))
