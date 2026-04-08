from torchvision import transforms
import pickle
import torch
from timesformer.models.vit import VisionTransformer
from functools import partial
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from dataset import PressSequenceDataset
from utils import denormalize_labels, SeqTransform

# =========================================
# Config
# =========================================
DATA_DIR   = "/home/data"
RESULT_DIR = "results"
BATCH_SIZE = 16
IMAGE_SIZE = 224
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["actual_TCP_pose_0", "ry_relative", "actual_TCP_pose_0_plus_dx_m"]

# Redirect logs to file
LOG_PATH = os.path.join(RESULT_DIR, "test_log.txt")
os.makedirs(RESULT_DIR, exist_ok=True)
sys.stdout = open(LOG_PATH, "w")

checkpoint_path = "checkpoints/Exp3"
checkpoint_name = "checkpoint_best"

# read hyperparameters and configuration
with open(os.path.join(checkpoint_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
f.close()
model_params = args["model_params"]
args["checkpoint_path"] = checkpoint_path
print(args)

## =========================================
# Load normalization parameters
# =========================================
stds, means = np.load(os.path.join(DATA_DIR, "norm_params.npy"))

# =========================================
# Dataset / DataLoader (TEST)
# =========================================
test_tf = SeqTransform(image_size=IMAGE_SIZE)
test_dataset = PressSequenceDataset(
    os.path.join(DATA_DIR, "test_img.npy"),
    os.path.join(DATA_DIR, "test_label.npy"),
    seq_transform=test_tf,
    normalize_labels=True, label_stds=stds, label_means=means
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True, persistent_workers=False
)

# build and load model
model = VisionTransformer(img_size=model_params["image_size"],
                          num_classes=model_params["num_classes"],
                          patch_size=model_params["patch_size"],
                          embed_dim=model_params["dim"],
                          depth=model_params["depth"],
                          num_heads=model_params["heads"],
                          mlp_ratio=4,
                          qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          drop_rate=0.,
                          attn_drop_rate=0.,
                          drop_path_rate=0.1,
                          num_frames=model_params["num_frames"],
                          attention_type=model_params["attention_type"])

checkpoint = torch.load(os.path.join(args["checkpoint_path"], "{}.pth".format(checkpoint_name)),
                        map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint['model_state_dict'])
if torch.cuda.is_available():
    model.cuda()

model.eval()

# =========================================
# Inference
# =========================================
preds_z, gts_z = [], []
with torch.no_grad():
    for seqs, ys in test_loader:
        seqs = seqs.to(DEVICE, non_blocking=True)
        outs = model(seqs)
        preds_z.append(outs.cpu().numpy())
        gts_z.append(ys.numpy())

preds_z = np.concatenate(preds_z, axis=0)
gts_z   = np.concatenate(gts_z,   axis=0)

# =========================================
# Denormalize
# =========================================
preds_dn = denormalize_labels(preds_z, stds, means)
gts_dn   = denormalize_labels(gts_z,   stds, means)


