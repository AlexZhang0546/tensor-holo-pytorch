# -*- coding: utf-8 -*-
"""Evaluate the OFFICIAL TensorHolo V2 stage-2 checkpoint (ddpm_12) on the
validation set using the CLEAN (NCHW) original code, matching the paper's
Table 2 protocol: post-DPM SSIM_amp/PSNR_amp + SSIM_img/PSNR_img."""
import sys, os, time
import numpy as np
import tensorflow as tf

TF_DIR = "/root/autodl-tmp/ZhangRuixuan/tensor-holo-clean"
DATA = "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch/data"
CKPT = "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch/ckpt_official_ddpm12"
sys.path.insert(0, TF_DIR)
import optics
import tfrecord
import main_v2 as main_v2_mod
from main_v2 import TensorHolographyModel


def main():
    dataset_res = 384
    pitch = 0.008
    hologram_params = {
        "wavelengths": np.array([0.000450, 0.000520, 0.000638]),
        "pitch": pitch, "res_h": dataset_res, "res_w": dataset_res,
        "depth_base": -3, "depth_scale": 6, "double_pad": True,
    }
    training_params = {
        "restore_trained_model": True, "batch": 2, "num_epochs": 4050,
        "decay_type": None, "decay_params": None, "learning_rate": 1e-4,
        "optimizer_type": "adam",
        "optimizer_params": {"beta1": 0.9, "beta2": 0.99, "epsilon": 1e-8},
        "num_iter_per_test": 1000, "num_top_depth_for_img_loss": 15,
        "num_random_depth_for_img_loss": 5, "depth_dependent_weight_scale": 0.35,
        "num_hist_bins": 200, "depth_shift": 12.0,
        "epoch_to_start_ddpm_training": 3000,
    }
    ddpm_params = {"active_max_ldi_layer": 0, "activate_ddpm": True,
                   "bypass_ddpm_network": False, "padding": 0}
    model_params = {
        "name": "full_loss", "input_dim": 4, "output_dim": 6,
        "num_layers": 30, "interleave_rate": 1, "num_filters_per_layer": 24,
        "filter_width": 3, "bias_stddev": 0.01, "weight_var_scale": 0.25,
        "renormalize_input": True, "activation_func": tf.nn.relu,
        "output_activation_func": tf.nn.tanh,
        "input_dim_ddpm": 6, "output_dim_ddpm": 6, "num_layers_ddpm": 8,
        "num_filters_per_layer_ddpm": 8, "filter_width_ddpm": 3,
        "interleave_rate_ddpm": 1, "bias_stddev_ddpm": 0.01,
        "weight_var_scale_ddpm": 0.25, "renormalize_input_ddpm": True,
        "activation_func_ddpm": tf.nn.relu,
        "output_activation_func_ddpm": tf.nn.tanh,
    }
    num_imgs_in_fs = 20
    loss_params = {
        "use_l2_loss": False,
        "loss_op": tf.compat.v1.losses.absolute_difference,
        "weight_holo": 1.0, "weight_fs": num_imgs_in_fs,
        "weight_fs_tv": num_imgs_in_fs, "weight_std": 0.02, "weight_mean": 0.03,
    }
    labels = ["amp_4", "phs_4", "img_0", "depth_0"]
    path_params = {
        "gen_record": False, "labels": labels,
        "train_output_path": os.path.join(DATA, "train_384_v2", "train_04.tfrecord"),
        "train_source_paths": [],
        "test_output_path": os.path.join(DATA, "test_384_v2", "test_04.tfrecord"),
        "test_source_paths": [],
        "validate_output_path": os.path.join(DATA, "validate_384_v2", "validate_04.tfrecord"),
        "validate_source_paths": [],
        "ckpt_path": os.path.join(CKPT, "ckpt"),
        "ckpt_parent_path": CKPT,
        "inference_graph_path": CKPT,
        "inference_graph_name": "inference_graph_v2",
    }

    def ds_params(batch):
        return {"repeat": True, "sample_count": 100, "batch": batch,
                "res_h": dataset_res, "res_w": dataset_res,
                "num_parallel_calls": 2, "prefetch_buffer_size": 4,
                "shuffle_buffer_size": 2, "num_epochs": 4050}
    train_dataset_params = dict(ds_params(2), sample_count=3800, num_parallel_calls=4)
    test_dataset_params = ds_params(2)
    validate_dataset_params = ds_params(2)

    model = TensorHolographyModel(
        hologram_params=hologram_params, training_params=training_params,
        ddpm_params=ddpm_params, model_params=model_params,
        loss_params=loss_params, path_params=path_params,
        train_dataset_params=train_dataset_params,
        test_dataset_params=test_dataset_params,
        validate_dataset_params=validate_dataset_params)
    # The original `_get_loss` references a bare `hologram_params` local that is
    # never assigned in the method body; bind it at module scope for evaluation.
    main_v2_mod.hologram_params = hologram_params

    (_, _, validate_handle, handle, rgbd, holo_in, amp_in, phs_in,
     holo_out, amp_out, phs_out) = model._setup_train()

    pad = ddpm_params["padding"]
    propagator_pad = optics.tf_propagator(
        (dataset_res + 2 * pad, dataset_res + 2 * pad), pitch,
        hologram_params["wavelengths"], method="as", double_pad=True)
    holo_in_s2, amp_in_s2, phs_in_s2, holo_out_s2, amp_out_s2, phs_out_s2, \
        amp_out_shifted_altered_s2, phs_out_shifted_altered_s2, \
        amp_out_shifted_s2, phs_out_shifted_s2 = model._setup_train_ddpm(
            holo_in, amp_in, phs_in, holo_out, amp_out, phs_out, propagator_pad)

    loss_s2, ssim_amp_s2, ssim_img_s2, psnr_amp_s2, psnr_img_s2, mean_s2, std_s2 = model._get_loss(
        holo_out_s2, amp_out_s2, phs_out_s2, holo_in_s2, amp_in_s2, phs_in_s2,
        rgbd, propagator_pad, phs_out_shifted_altered_s2)

    model.saver = tf.compat.v1.train.Saver(max_to_keep=5, save_relative_paths=True)
    model.sess.run(tf.compat.v1.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(CKPT)
    if ckpt is None or not ckpt.model_checkpoint_path:
        raise RuntimeError("no checkpoint in " + CKPT)
    print("restoring from", ckpt.model_checkpoint_path, flush=True)
    model.saver.restore(model.sess, ckpt.model_checkpoint_path)

    n_steps = validate_dataset_params["sample_count"] // validate_dataset_params["batch"]
    vals = {k: [] for k in ["ssim_amp", "ssim_img", "psnr_amp", "psnr_img", "mean", "std"]}
    t0 = time.time()
    for step in range(n_steps):
        ra, ri, pa, pi, mn, st = model.sess.run(
            [ssim_amp_s2, ssim_img_s2, psnr_amp_s2, psnr_img_s2, mean_s2, std_s2],
            feed_dict={handle: validate_handle})
        vals["ssim_amp"].append(ra)
        vals["ssim_img"].append(ri)
        vals["psnr_amp"].append(pa)
        vals["psnr_img"].append(pi)
        vals["mean"].append(mn)
        vals["std"].append(st)
        if step % 5 == 0:
            print("step %d/%d (%.0fs)" % (step, n_steps, time.time() - t0), flush=True)
    for k in ["ssim_amp", "ssim_img", "psnr_amp", "psnr_img", "mean", "std"]:
        print("%s: mean %.4f  std %.4f  max %.4f  min %.4f" % (
            k, np.mean(vals[k]), np.std(vals[k]), np.max(vals[k]), np.min(vals[k])), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
