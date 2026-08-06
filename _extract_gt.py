import sys
import os
import torch
import numpy as np

sys.path.insert(0, ".")
from src.data.dataset import THDataset
from src.optics.complex_utils import compl_val

ds = THDataset(
    "data/test_384_v2/test_04.tfrecord",
    {"res_h": 384, "res_w": 384, "sample_count": 100},
    ["amp_4", "phs_4", "img_0", "depth_0"], 0, True)
b = next(iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)))
amp_gt = b["amp_4"].numpy()[0]          # (3,384,384)
phs_gt = b["phs_4"].numpy()[0]          # [0,1]
holo_gt = compl_val(torch.from_numpy(amp_gt),
                    (torch.from_numpy(phs_gt) - 0.5) * 2.0 * np.pi).numpy()

out_dir = "_gt_dump"
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "amp_gt.npy"), amp_gt)
np.save(os.path.join(out_dir, "phs_gt.npy"), phs_gt)
np.save(os.path.join(out_dir, "holo_gt.npy"), holo_gt)
print("saved", amp_gt.shape, phs_gt.shape, holo_gt.shape, holo_gt.dtype)
