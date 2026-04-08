
# ====================== custom_collate (DinoVO용) ======================
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
import time
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from functools import partial
import pickle
import torch
from train_Convnext_vit import VideoConvNeXtViT
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
from train_eff_vit import VideoEfficientViT
import timm

# =========================================
# Config
# =========================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/home/data"
RESULT_DIR = "results"
BATCH_SIZE = 16
IMAGE_SIZE = 224
LABEL_NAMES = ["actual_TCP_pose_0", "ry_relative", "actual_TCP_pose_0_plus_dx_m"]

# 로그 리다이렉트
LOG_PATH = os.path.join(RESULT_DIR, "test_log.txt")
os.makedirs(RESULT_DIR, exist_ok=True)
sys.stdout = open(LOG_PATH, "w")
print(f"Using device: {DEVICE}")

# =========================================
# Checkpoint에서 하이퍼파라미터 로드
# =========================================
checkpoint_path = "checkpoints/Exp3"      # ← 필요시 경로 수정
checkpoint_name = "checkpoint_best"

with open(os.path.join(checkpoint_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)

model_params = args["model_params"]
args["checkpoint_path"] = checkpoint_path
print("Loaded args:", args)

# =========================================
# Load normalization parameters
# =========================================
stds, means = np.load(os.path.join(DATA_DIR, "norm_params_cbts.npy"))   # ← TSformer용 norm 파일명 확인

# =========================================
# Test Dataset & DataLoader
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

# =========================================
# Build TSformer Model
# =========================================
# build and load model
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



# =========================================
# Load Checkpoint
# =========================================
checkpoint = torch.load(
    os.path.join(args["checkpoint_path"], f"{checkpoint_name}.pth"),
    map_location=DEVICE
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

print(f"✅ TSformer Model loaded from {checkpoint_name}.pth")

# =========================================
# Peak Memory & Time Measurement
# =========================================
torch.cuda.reset_peak_memory_stats(DEVICE)
torch.cuda.empty_cache()
start_time = time.time()

# =========================================
# Inference
# =========================================
preds_z = []
gts_z = []

with torch.no_grad():
    for seqs, ys in tqdm(test_loader, desc="[Test]"):
        seqs = seqs.to(DEVICE, non_blocking=True)
        # ys는 이미 tensor로 로드됨
        outs = model(seqs)                    # TSformer forward: (B, num_frames, C, H, W) → (B, 3)

        preds_z.append(outs.cpu().numpy())
        gts_z.append(ys.cpu().numpy())        # label은 3차원이라고 가정

preds_z = np.concatenate(preds_z, axis=0)
gts_z = np.concatenate(gts_z, axis=0)

print(f"Inference completed! Pred shape: {preds_z.shape}, GT shape: {gts_z.shape}")

# =========================================
# Denormalize
# =========================================
preds_dn = denormalize_labels(preds_z, stds, means)
gts_dn = denormalize_labels(gts_z, stds, means)

# =========================================
# Save Predictions with TOTAL row
# =========================================
pred_csv = os.path.join(RESULT_DIR, "predictions_test.csv")

df_pred = pd.DataFrame({
    f"{LABEL_NAMES[0]}_gt": gts_dn[:, 0],
    f"{LABEL_NAMES[0]}_pred": preds_dn[:, 0],
    f"{LABEL_NAMES[1]}_gt": gts_dn[:, 1],
    f"{LABEL_NAMES[1]}_pred": preds_dn[:, 1],
    f"{LABEL_NAMES[2]}_gt": gts_dn[:, 2],
    f"{LABEL_NAMES[2]}_pred": preds_dn[:, 2],
})

# TOTAL 행 추가
total_row = pd.DataFrame([{
    f"{LABEL_NAMES[0]}_gt": gts_dn[:, 0].sum(),
    f"{LABEL_NAMES[0]}_pred": preds_dn[:, 0].sum(),
    f"{LABEL_NAMES[1]}_gt": gts_dn[:, 1].sum(),
    f"{LABEL_NAMES[1]}_pred": preds_dn[:, 1].sum(),
    f"{LABEL_NAMES[2]}_gt": gts_dn[:, 2].sum(),
    f"{LABEL_NAMES[2]}_pred": preds_dn[:, 2].sum(),
}])

df_pred = pd.concat([df_pred, total_row], ignore_index=True)
df_pred.to_csv(pred_csv, index=False)

print(f"✅ predictions_test.csv saved with TOTAL row → {len(df_pred)} rows (including TOTAL)")

# =========================================
# Metrics
# =========================================
print("\n=== Test Metrics (z-score space) ===")
for i, name in enumerate(LABEL_NAMES):
    mae = mean_absolute_error(gts_z[:, i], preds_z[:, i])
    rmse = np.sqrt(mean_squared_error(gts_z[:, i], preds_z[:, i]))
    print(f"{name}: MAE={mae:.6f}, RMSE={rmse:.6f}")

print("\n=== Test Metrics (denormalized, real units) ===")
for i, name in enumerate(LABEL_NAMES):
    mae = mean_absolute_error(gts_dn[:, i], preds_dn[:, i])
    rmse = np.sqrt(mean_squared_error(gts_dn[:, i], preds_dn[:, i]))
    print(f"{name}: MAE={mae:.6f}, RMSE={rmse:.6f}")

# =========================================
# Final Summary
# =========================================
end_time = time.time()
total_time = end_time - start_time
peak_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
reserved_mb = torch.cuda.max_memory_reserved(DEVICE) / (1024 ** 2)

summary = f"""
{'='*70}
✅ TSformer Inference Completed Successfully!
Device          : {DEVICE}
Total Time      : {total_time:.2f} seconds
Batch Size      : {BATCH_SIZE}
Number of Batches : {len(test_loader)}
Final Peak Memory : {peak_mb:.1f} MB
Max Reserved Memory : {reserved_mb:.1f} MB
{'='*70}
"""
print(summary)

# 메모리 정리
torch.cuda.empty_cache()
gc.collect()
print(f"\nResults saved to: {RESULT_DIR}")