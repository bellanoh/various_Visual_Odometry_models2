
import random
from tqdm import tqdm

import os
import pickle
import torch
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from dataset import PressSequenceDataset
from utils import denormalize_labels, SeqTransform
import timm
from train_Convnext_vit import VideoConvNeXtViT

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

#
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
base_model = timm.create_model(
    'convnext_tiny.fb_in1k',  # ConvNeXt-Tiny (dim≈768, heads-like 구조, depth≈52 blocks but effective 12-like)
    pretrained=True,  # pretrained weights 로드
    num_classes=model_params["num_classes"],  # 3 (VO용 3 DoF – 6 DoF면 6으로)
    drop_rate=model_params["dropout"],  # 0.6
    drop_path_rate=model_params["ff_dropout"],  # 0.2 (ConvNeXt의 stochastic depth ≈ ff_dropout)
)

model = VideoConvNeXtViT(
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
NOISE_PROB = 0.3      # 노이즈 적용 확률 (0~1)
NOISE_STD = 0.08
# =========================================
# Inference
# =========================================
preds_z, gts_z = [], []
with torch.no_grad():
    for batch_idx, (seqs, ys) in enumerate(tqdm(test_loader, desc="[Test + Noise]")):

        seqs = seqs.to(DEVICE, non_blocking=True)
        ys = ys.to(DEVICE, non_blocking=False)

        # ====================== Gaussian Noise 추가 ======================
        if random.random() < NOISE_PROB:
            # seqs 전체에 Gaussian Noise 추가
            noise = torch.randn_like(seqs) * NOISE_STD
            seqs = torch.clamp(seqs + noise, 0.0, 1.0)  # 값 범위 유지 (중요!)

            if (batch_idx + 1) % 20 == 0:
                print(f"Batch {batch_idx + 1:4d}: Gaussian Noise added (std={NOISE_STD})")

        # ====================== Model Inference ======================
        outs = model(seqs)

        preds_z.append(outs.cpu().numpy())
        gts_z.append(ys.cpu().numpy())  # 또는 ys[:, :3].cpu().numpy() 필요시

# Concatenate
preds_z = np.concatenate(preds_z, axis=0)
gts_z = np.concatenate(gts_z, axis=0)
# =========================================
# Denormalize
# =========================================
preds_dn = denormalize_labels(preds_z, stds, means)
gts_dn   = denormalize_labels(gts_z,   stds, means)

# # =========================================
# # Evaluation Metrics
# # =========================================
print("\n=== Test Metrics (z-score space) ===")
rows_z = []
for i, name in enumerate(LABEL_NAMES):
    mae  = mean_absolute_error(gts_z[:, i], preds_z[:, i])
    mse  = mean_squared_error(gts_z[:, i], preds_z[:, i])
    rmse = np.sqrt(mse)
    print(f"{name}: MAE={mae:.6f}, RMSE={rmse:.6f}")
    rows_z.append({"label": name, "MAE_z": mae, "RMSE_z": rmse})

print("\n=== Test Metrics (denormalized, real units) ===")
rows_dn = []
for i, name in enumerate(LABEL_NAMES):
    mae  = mean_absolute_error(gts_dn[:, i], preds_dn[:, i])
    mse  = mean_squared_error(gts_dn[:, i], preds_dn[:, i])
    rmse = np.sqrt(mse)
    print(f"{name}: MAE={mae:.6f}, RMSE={rmse:.6f}")
    rows_dn.append({"label": name, "MAE_dn": mae, "RMSE_dn": rmse})

# ========= Save CSVs =========
pred_csv = os.path.join(RESULT_DIR, "predictions_test.csv")
pd.DataFrame({
    f"{LABEL_NAMES[0]}_gt": gts_dn[:, 0], f"{LABEL_NAMES[0]}_pred": preds_dn[:, 0],
    f"{LABEL_NAMES[1]}_gt": gts_dn[:, 1], f"{LABEL_NAMES[1]}_pred": preds_dn[:, 1],
    f"{LABEL_NAMES[2]}_gt": gts_dn[:, 2], f"{LABEL_NAMES[2]}_pred": preds_dn[:, 2],
}).to_csv(pred_csv, index=False)

metric_csv = os.path.join(RESULT_DIR, "metrics_test.csv")
pd.concat([pd.DataFrame(rows_z), pd.DataFrame(rows_dn)], axis=1).to_csv(metric_csv, index=False)
print(f"\nSaved files:\n- {pred_csv}\n- {metric_csv}")
