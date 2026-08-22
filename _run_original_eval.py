import os
import sys
import runpy

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.layers = tf.compat.v1.layers
tf.losses = tf.compat.v1.losses


def _depth_to_space_nchw(x, block_size):
    """Replace DepthToSpace for NCHW, which newer CPU TF builds lack."""
    n = tf.shape(x)[0]
    c4 = tf.shape(x)[1]
    h = tf.shape(x)[2]
    w = tf.shape(x)[3]
    c = c4 // (block_size * block_size)
    x = tf.reshape(x, [n, c, block_size, block_size, h, w])
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    return tf.reshape(x, [n, c, h * block_size, w * block_size])


_tf_depth_to_space = tf.compat.v1.depth_to_space


def _patched_depth_to_space(x, block_size, data_format="NHWC"):
    if data_format == "NCHW":
        return _depth_to_space_nchw(x, block_size)
    return _tf_depth_to_space(x, block_size, data_format=data_format)


tf.compat.v1.depth_to_space = _patched_depth_to_space
tf.depth_to_space = _patched_depth_to_space


_tf_nn_conv2d = tf.nn.conv2d
_tf_nn_bias_add = tf.nn.bias_add
_tf_nn_depthwise_conv2d = tf.nn.depthwise_conv2d


def _nhwc_strides(strides):
    # NCHW [1,1,h,w] -> NHWC [1,h,w,1]
    return [strides[0], strides[2], strides[3], strides[1]]


def _conv2d_cpu(input, filter, strides, padding, data_format="NHWC", **kwargs):
    if data_format == "NCHW":
        x = tf.transpose(input, [0, 2, 3, 1])
        y = _tf_nn_conv2d(x, filter, _nhwc_strides(strides), padding,
                          data_format="NHWC", **kwargs)
        return tf.transpose(y, [0, 3, 1, 2])
    return _tf_nn_conv2d(input, filter, strides, padding,
                         data_format=data_format, **kwargs)


def _bias_add_cpu(value, bias, data_format="NHWC", **kwargs):
    if data_format == "NCHW":
        x = tf.transpose(value, [0, 2, 3, 1])
        y = _tf_nn_bias_add(x, bias, data_format="NHWC", **kwargs)
        return tf.transpose(y, [0, 3, 1, 2])
    return _tf_nn_bias_add(value, bias, data_format=data_format, **kwargs)


def _depthwise_conv2d_cpu(input, filter, strides, padding, data_format="NHWC",
                          **kwargs):
    if data_format == "NCHW":
        x = tf.transpose(input, [0, 2, 3, 1])
        y = _tf_nn_depthwise_conv2d(x, filter, _nhwc_strides(strides), padding,
                                    data_format="NHWC", **kwargs)
        return tf.transpose(y, [0, 3, 1, 2])
    return _tf_nn_depthwise_conv2d(input, filter, strides, padding,
                                   data_format=data_format, **kwargs)


tf.nn.conv2d = _conv2d_cpu
tf.nn.bias_add = _bias_add_cpu
tf.nn.depthwise_conv2d = _depthwise_conv2d_cpu

if __name__ == "__main__":
    # argparse in main_v2.py sees sys.argv[1:]; keep script name as argv[0].
    original_script = sys.argv[1]
    original_args = sys.argv[2:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(original_script)))
    sys.argv = [original_script] + original_args
    runpy.run_path(original_script, run_name="__main__")
