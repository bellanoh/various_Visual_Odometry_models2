
import warnings

warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated"
)
import pickle
import torch
from timesformer.models.vit import VisionTransformer
import torch
import torch.nn as nn
from functools import partial
import time
import numpy as np
import timm
from train_Convnext_vit import VideoConvNeXtViT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
T = 20

print(f"Device: {DEVICE}")

# ====================== TSformer 모델 생성 ======================
# build and load model
base_model = timm.create_model(
    "convnext_tiny.fb_in1k",  # ConvNeXt-Tiny (dim≈768, heads-like 구조, depth≈52 blocks but effective 12-like)
    pretrained=True,  # pretrained weights 로드
    num_classes=3,  # 3 (VO용 3 DoF – 6 DoF면 6으로)
    drop_rate=0.,  # 0.6
    drop_path_rate=0.,  # 0.2 (ConvNeXt의 stochastic depth ≈ ff_dropout)
)

model = VideoConvNeXtViT(
    base_model,
    num_frames=20,
    attention_type="divided_space_time",
    attn_dropout=0.,
    dim_head=128,
    heads=6,
    time_only=False
)
model = model.to(DEVICE)
model.eval()
print("✅ TSformer model created successfully!\n")

# ====================== 더미 데이터 ======================
dummy_input = torch.randn(BATCH_SIZE, T, 1, 224, 224, device=DEVICE)

# Warmup
with torch.no_grad():
    for _ in range(20):
        _ = model(dummy_input)
torch.cuda.synchronize()

# ====================== FPS 측정 ======================
times = []
with torch.no_grad():
    for _ in range(150):
        start = time.perf_counter()
        _ = model(dummy_input)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

avg_time = np.mean(times)
fps_batch = 1.0 / avg_time

print("=" * 80)
print("TSformer FPS Measurement Result")
print("=" * 80)
print(f"Average time per batch     : {avg_time*1000:.2f} ms")
print(f"FPS (batches/sec)          : {fps_batch:.2f}")
print(f"Images per second          : {fps_batch * BATCH_SIZE * T:.1f}")
print("=" * 80)

import torch
state_dict = torch.load("results/best_model.pth", map_location="cpu", weights_only=True)

total_params = sum(p.numel() for p in state_dict.values())
print(f"Total parameters in .pth file: {total_params:,}")
print(f"Model size ≈ {total_params / 1_000_000:.2f} M parameters")
print(f"Approx. file size (float32): {total_params * 4 / (1024**2):.2f} MB")

