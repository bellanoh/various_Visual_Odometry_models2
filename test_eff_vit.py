from torchvision import transforms
import pickle
import torch
#from timesformer.models.vit import VisionTransformer
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
import timm
from train_eff_vit import VideoEfficientViT
# =========================================
# Config
# =========================================

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
base_model =timm.create_model(
        'efficientvit_b2.r224_in1k', #b1은 dim 더 작음
        pretrained=True,  # pretrained weights 로드 (ImageNet-1K 기반)
        num_classes=model_params["num_classes"],  # 3 (VO용 3 DoF – 6 DoF면 6으로)
        drop_rate=model_params["dropout"],  # 0.6
    )


model = VideoEfficientViT(
        base_model,
        num_frames=model_params["num_frames"],
        attention_type=model_params["attention_type"],
        attn_dropout=model_params["attn_dropout"],
        dim_head=model_params["dim_head"],
        heads=model_params["heads"],
        time_only=model_params["time_only"]
    )

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

