import tensorflow as tf
r = tf.train.NewCheckpointReader("model/ckpt_full_loss_pitch_8_layers_30_filters_24_ldi_0_ddpm_12/ckpt-7695000")
names = r.get_variable_to_shape_map()
for k in sorted(names):
    if "ddpm" in k.lower():
        print(k, names[k])
