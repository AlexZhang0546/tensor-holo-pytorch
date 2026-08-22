import os
import sys
import runpy

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.layers = tf.compat.v1.layers
tf.losses = tf.compat.v1.losses

if __name__ == "__main__":
    # argparse in main_v2.py sees sys.argv[1:]; keep script name as argv[0].
    original_script = sys.argv[1]
    original_args = sys.argv[2:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(original_script)))
    sys.argv = [original_script] + original_args
    runpy.run_path(original_script, run_name="__main__")
