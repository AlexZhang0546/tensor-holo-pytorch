# -*- coding: utf-8 -*-
"""Compute PRE-DPM (stage-1 style) metrics for the OFFICIAL ddpm_12 checkpoint's
main CNN using the official graph. Answers: is the paper's 0.945 measured on the
joint-trained main CNN output?"""
import sys, os, time
import numpy as np
import tensorflow as tf

TF_DIR = "/root/autodl-tmp/ZhangRuixuan/tensor-holo-official"
DATA = os.path.join(TF_DIR, "data")
CKPT_PARENT = os.path.join(TF_DIR, "model", "ckpt_full_loss_pitch_8_layers_30_filters_24_ldi_0_ddpm_12")
sys.path.insert(0, TF_DIR)
import optics
import tfrecord
from main_v2 import TensorHolographyModel

def main():
    dataset_res = 384
    pitch = 0.008
    hologram_params = {"wavelengths": np.array([0.000450, 0.000520, 0.000638]),
        "pitch": pitch, "res_h": dataset_res, "res_w": dataset_res,
        "depth_base": -3, "depth_scale": 6, "double_pad": True}
    training_params = {"restore_trained_model": True, "batch": 2, "num_epochs": 4050,
        "decay_type": None, "decay_params": None, "learning_rate": 1e-4,
        "optimizer_type": "adam", "optimizer_params": {"beta1":0.9,"beta2":0.99,"epsilon":1e-8},
        "num_iter_per_test": 1000, "num_top_depth_for_img_loss": 15,
        "num_random_depth_for_img_loss": 5, "depth_dependent_weight_scale": 0.35,
        "num_hist_bins": 200, "depth_shift": 12.0, "epoch_to_start_ddpm_training": 3000}
    ddpm_params = {"active_max_ldi_layer": 0, "activate_ddpm": True,
                   "bypass_ddpm_network": False, "padding": 0}
    model_params = {"name":"full_loss","input_dim":4,"output_dim":6,"num_layers":30,
        "interleave_rate":1,"num_filters_per_layer":24,"filter_width":3,
        "bias_stddev":0.01,"weight_var_scale":0.25,"renormalize_input":True,
        "activation_func":tf.nn.relu,"output_activation_func":tf.nn.tanh,
        "input_dim_ddpm":6,"output_dim_ddpm":6,"num_layers_ddpm":8,
        "num_filters_per_layer_ddpm":8,"filter_width_ddpm":3,"interleave_rate_ddpm":1,
        "bias_stddev_ddpm":0.01,"weight_var_scale_ddpm":0.25,"renormalize_input_ddpm":True,
        "activation_func_ddpm":tf.nn.relu,"output_activation_func_ddpm":tf.nn.tanh}
    num_imgs_in_fs = 20
    loss_params = {"use_l2_loss":False,"loss_op":tf.compat.v1.losses.absolute_difference,
        "weight_holo":1.0,"weight_fs":num_imgs_in_fs,"weight_fs_tv":num_imgs_in_fs,
        "weight_std":0.02,"weight_mean":0.03}
    labels = ["amp_4","phs_4","img_0","depth_0"]
    path_params = {"gen_record":False,"labels":labels,
        "train_output_path":os.path.join(DATA,"train_384_v2","train_04.tfrecord"),
        "train_source_paths":[],"test_output_path":os.path.join(DATA,"test_384_v2","test_04.tfrecord"),
        "test_source_paths":[],"validate_output_path":os.path.join(DATA,"validate_384_v2","validate_04.tfrecord"),
        "validate_source_paths":[],"ckpt_path":os.path.join(CKPT_PARENT,"ckpt"),
        "ckpt_parent_path":CKPT_PARENT,"inference_graph_path":CKPT_PARENT,
        "inference_graph_name":"inference_graph_v2"}
    def ds_params(batch):
        return {"repeat":True,"sample_count":100,"batch":batch,"res_h":dataset_res,
                "res_w":dataset_res,"num_parallel_calls":2,"prefetch_buffer_size":4,
                "shuffle_buffer_size":2,"num_epochs":4050}
    train_dataset_params = dict(ds_params(2), sample_count=3800, num_parallel_calls=4)
    test_dataset_params = ds_params(2)
    validate_dataset_params = ds_params(2)

    model = TensorHolographyModel(hologram_params=hologram_params, training_params=training_params,
        ddpm_params=ddpm_params, model_params=model_params, loss_params=loss_params,
        path_params=path_params, train_dataset_params=train_dataset_params,
        test_dataset_params=test_dataset_params, validate_dataset_params=validate_dataset_params)

    train_handle, test_handle, validate_handle, handle, rgbd, holo_in, amp_in, phs_in, holo_out, amp_out, phs_out = model._setup_train()
    propagator_pad = optics.tf_propagator((dataset_res, dataset_res), pitch,
        hologram_params["wavelengths"], method="as", double_pad=True)
    global_step = tf.Variable(0, trainable=False)
    optimizer = model._setup_optimizer(starter_learning_rate=1e-4, decay_type=None,
        decay_params=None, opt_type="adam", opt_params={"beta1":0.9,"beta2":0.99,"epsilon":1e-8},
        global_step=global_step)
    holo_in_s2, amp_in_s2, phs_in_s2, holo_out_s2, amp_out_s2, phs_out_s2, \
        amp_out_shifted_altered_s2, phs_out_shifted_altered_s2, amp_out_shifted_s2, phs_out_shifted_s2 = \
        model._setup_train_ddpm(holo_in, amp_in, phs_in, holo_out, amp_out, phs_out, propagator_pad)
    loss_s2, ssim_amp_s2, ssim_img_s2, psnr_amp_s2, psnr_img_s2, mean_s2, std_s2 = model._get_loss(
        holo_out_s2, amp_out_s2, phs_out_s2, holo_in_s2, amp_in_s2, phs_in_s2,
        rgbd, propagator_pad, phs_out_shifted_altered_s2)
    # pre-DPM stage-1 metrics using the SAME main CNN weights
    loss_s1, ssim_amp_pre, ssim_img_pre, psnr_amp_pre, psnr_img_pre, _, _ = model._get_loss(
        holo_out, amp_out, phs_out, holo_in, amp_in, phs_in, rgbd, propagator_pad, None)
    optimizer.minimize(loss=loss_s2, global_step=global_step)

    ckpt_path = os.path.join(CKPT_PARENT, "ckpt-7695000")
    print("restoring", ckpt_path, flush=True)
    reader = tf.train.NewCheckpointReader(ckpt_path)
    ckpt_names = set(reader.get_variable_to_shape_map().keys())
    gv = {v.name[:-2]: v for v in tf.compat.v1.global_variables()}
    restore_map = {n: v for n, v in gv.items() if n in ckpt_names}
    print("graph vars: %d ckpt vars: %d restoring: %d" % (len(gv), len(ckpt_names), len(restore_map)), flush=True)
    model.sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(var_list=restore_map)
    saver.restore(model.sess, ckpt_path)

    n_steps = validate_dataset_params["sample_count"] // training_params["batch"]
    vals = {k: [] for k in ["pre_amp","pre_img","post_amp","post_img","pre_psnr_amp","pre_psnr_img"]}
    t0 = time.time()
    for step in range(n_steps):
        ra, ri, pa, pi, ppa, ppi = model.sess.run(
            [ssim_amp_pre, ssim_img_pre, ssim_amp_s2, ssim_img_s2, psnr_amp_pre, psnr_img_pre],
            feed_dict={handle: validate_handle})
        vals["pre_amp"].append(ra); vals["pre_img"].append(ri)
        vals["post_amp"].append(pa); vals["post_img"].append(pi)
        vals["pre_psnr_amp"].append(ppa); vals["pre_psnr_img"].append(ppi)
        if step % 10 == 0:
            print("step %d/%d (%.0fs)" % (step, n_steps, time.time()-t0), flush=True)
    for k in ["pre_amp","pre_img","post_amp","post_img","pre_psnr_amp","pre_psnr_img"]:
        a = np.array(vals[k])
        print("%s: mean %.4f  std %.4f  max %.4f  min %.4f" % (k, a.mean(), a.std(), a.max(), a.min()), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()