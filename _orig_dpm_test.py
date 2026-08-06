import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

ORIG_DIR = "/root/autodl-tmp/tensor_holography"
sys.path.insert(0, ORIG_DIR)
import optics

DUMP = "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch/_gt_dump"
amp_gt = np.load(os.path.join(DUMP, "amp_gt.npy"))
phs_gt = np.load(os.path.join(DUMP, "phs_gt.npy"))
holo_gt = np.load(os.path.join(DUMP, "holo_gt.npy"))

wavelengths = np.array([0.000450, 0.000520, 0.000638])
pitch = 0.008
depth_shift = 12.0

propagator = optics.tf_propagator((384, 384), pitch, wavelengths,
                                  method="as", double_pad=True)

holo_gt_t = tf.constant(holo_gt[None], dtype=tf.complex64)      # (1,3,384,384)
amp_gt_t = tf.constant(amp_gt[None], dtype=tf.float32)

# shift to hologram plane
tf_wavelength = tf.constant(wavelengths.reshape(1, 3, 1, 1))
holo_shift = propagator(holo_gt_t, depth_shift) * optics.tf_compl_exp(
    -2 * np.pi * depth_shift / tf_wavelength)

# aadpm (same config as stage2)
phs_only, amp_max = optics.tf_aadpm(
    holo_shift, propagator, depth_shift=0, adaptive_phs_shift=False,
    batch=1, num_channels=3, res_h=384, res_w=384, sigma=0.0,
    kernel_width=3, phs_max=None, amp_max=None, clamp=True,
    normalize=False, wavelength=wavelengths.tolist())

# filter phase only
amp_out, phs_out = optics.tf_filter_phs_only(
    phs_only, unnormalize_input=False, normalize_output=False,
    propagator=propagator, depth_shift=-depth_shift, batch=1,
    num_channels=3, res_h=384, res_w=384, radius=None, phs_max=None,
    amp_max=amp_max, wavelength=wavelengths.tolist())

holo_out = optics.tf_compl_val(amp_out, phs_out)

ssim_amp = tf.reduce_mean(tf.image.ssim(
    tf.transpose(amp_out, [0, 2, 3, 1]),
    tf.transpose(amp_gt_t, [0, 2, 3, 1]), 1.0))

# 导出中间量用于与端口逐元素对照
save_ops = {
    "orig_holo_shift": holo_shift,
    "orig_phs_only": phs_only,
    "orig_amp_out": amp_out,
    "orig_phs_out": phs_out,
}

# focal-plane image comparison
ssim_focal = {}
for focus in [-3.0, 0.0, 3.0]:
    img_gt = tf.abs(propagator(holo_gt_t, -focus))
    img_out = tf.abs(propagator(holo_out, -focus))
    ssim_focal[focus] = tf.reduce_mean(tf.image.ssim(
        tf.transpose(img_gt, [0, 2, 3, 1]),
        tf.transpose(img_out, [0, 2, 3, 1]), 1.0))

with tf.Session() as sess:
    amp_max_val = sess.run(amp_max)
    print("orig amp_max:", np.round(np.asarray(amp_max_val).flatten(), 4).tolist())
    s = sess.run(ssim_amp)
    print("orig SSIM(amp_out, amp_gt) = %.4f" % s)
    for focus, op in ssim_focal.items():
        print("orig focal focus %+4.1fmm SSIM = %.4f" % (focus, sess.run(op)))

    # 输出 phs_only 范围
    po = sess.run(phs_only)
    print("orig phs_only range: %.3f .. %.3f" % (po.min(), po.max()))
    for name, op in save_ops.items():
        val = sess.run(op)
        np.save(os.path.join(DUMP, name + ".npy"), val[0])
        print("saved", name, val.shape, val.dtype)
