import sys
import os

sys.path.insert(0, ".")
import main

cur = os.path.dirname(os.path.realpath("main.py"))
print("modules imported OK")

cases = [
    (["--train-mode", "--train-stage", "stage1", "--dataset-res", "384"], "train_stage1"),
    (["--train-mode", "--train-stage", "stage2", "--dataset-res", "384", "--activate-ddpm",
      "--restore-stage1", "--train-depth-shift", "12.0", "--epoch_to_start_ddpm_training", "0",
      "--stage1-ckpt", "/tmp/s1.pth", "--restore-stage2", "--stage2-ckpt-dir", "/tmp/s2"], "train_stage2"),
    (["--validate-mode-s1", "--dataset-res", "384"], "validate-s1"),
    (["--validate-mode-s2", "--dataset-res", "384", "--activate-ddpm", "--batch", "1"], "validate-s2"),
    (["--eval-mode", "--ckpt-path", "/tmp/s1.pth", "--eval-rgb-path", "/tmp/a.png",
      "--eval-depth-path", "/tmp/b.png", "--eval-output-path", "/tmp/out", "--phs-max", "2.0"], "evaluate"),
    (["--export-mode", "--trt-res-h", "1080", "--trt-res-w", "1920", "--activate-ddpm"], "export"),
]

for argv, label in cases:
    args = main.build_original_parser().parse_args(argv)
    if label == "train_stage1":
        out = main._build_stage1_argv(args)
    elif label == "train_stage2":
        out = main._build_stage2_argv(args, cur)
    elif label == "validate-s1":
        out = main._build_validate_argv(args, "stage1", cur)
    elif label == "validate-s2":
        out = main._build_validate_argv(args, "stage2", cur)
    elif label == "evaluate":
        ns = main._build_eval_namespace(args, cur)
        print("== evaluate (namespace)")
        print("ckpt_path:", ns.ckpt_path)
        print("phs_max:", ns.phs_max, "pitch:", ns.pitch)
        continue
    else:
        out = main._build_export_argv(args, cur)
    print("== %s" % label)
    print(" ".join(out))
    print()
