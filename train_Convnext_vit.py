
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.optim as optim
# from build_model import build_model
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import pickle
import json
import torch.optim.lr_scheduler as schedulers
import math
import timm

from dataset import PressSequenceDataset
from utils import SeqTransform, save_checkpoint_full, denormalize_labels

# =========================================
# Setup
# =========================================

# =========================================
# Normalization (from TRAIN labels)
# =========================================
stds, means = np.load(os.path.join(DATA_DIR, "norm_params_cbts.npy"))

# =========================================
# Transforms
# =========================================
train_tf = SeqTransform(image_size=IMAGE_SIZE)
val_tf   = SeqTransform(image_size=IMAGE_SIZE)

# =========================================
# Datasets
# =========================================
train_dataset = PressSequenceDataset(
    os.path.join(DATA_DIR, "train_img.npy"),
    os.path.join(DATA_DIR, "train_label.npy"),
    seq_transform=train_tf,
    normalize_labels=True, label_stds=stds, label_means=means
)
val_dataset = PressSequenceDataset(
    os.path.join(DATA_DIR, "val_img.npy"),
    os.path.join(DATA_DIR, "val_label.npy"),
    seq_transform=val_tf,
    normalize_labels=True, label_stds=stds, label_means=means
)


# =========================================
# Custom model
# =========================================
class VideoConvNeXtViT(torch.nn.Module):
    def __init__(self, base_model, num_frames, attention_type='divided_space_time', attn_dropout=0.1, dim_head=64,
                 heads=6, time_only=False):
        super().__init__()
        self.base_model = base_model
        self.num_frames = num_frames  # 4 (from args["window_size"])
        self.attention_type = attention_type  # 'divided_space_time'
        self.time_only = time_only  # False
        self.feature_extractor = lambda x: self.base_model.forward_head(
            self.base_model.forward_features(x), pre_logits=True
        ) #ConvNeXt Tiny의 경우 출력 모양은 (B*T, 768, 7, 7)입니다.
        embed_dim = base_model.num_features  # ConvNeXt embed dim (e.g., 384 for Tiny)
        # Time attention 추가 (ViT-like: MultiheadAttention)
        self.time_attn = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=attn_dropout,)
                                                     #kdim=dim_head, vdim=dim_head)

    def forward(self, x):
        # x: [batch, num_frames, 3, H, W] – 비디오 입력 가정
        """
                  x: (B, T, C, H, W)
                  """
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        if C == 1:
            x = x.repeat(1, 3, 1, 1)  # grayscale → RGB

        feats = self.feature_extractor(x)  # (B*T, 512)
        feats = feats.reshape(B, T, -1)  # (B, T, 512) 768 * 7 * 7 = 37632가 되지 않게 위에

        # attention_type 적용 (divided_space_time: space는 ConvNeXt, time은 별도 attention)
        if self.attention_type == 'divided_space_time' and not self.time_only:
            feats = feats.transpose(0, 1)  # (T, B, 768)
            feats = self.time_attn(feats, feats, feats)[0]
            feats = feats.transpose(0, 1)  # (B, T, 768)
        # 다른 type 구현: space_only (time_attn skip), joint_space_time (공동), time_only (space skip)
        feats = feats.mean(1)  # (B, 768)
        # Dropout 적용 (head의 drop_rate 확인; pretrained 모델은 보통 0임)
        feats = self.base_model.head.drop(feats)

        out = self.base_model.head.fc(feats)  # fc만 호출: (B, 768) -> (B, 3)
        return out


def val_epoch(model, val_loader, criterion, args):
    epoch_loss = 0
    with tqdm(val_loader, unit="batch") as tepoch:
        for images, gt in tepoch:
            tepoch.set_description(f"Validating ")
            # for batch_idx, (images, odom) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

            # predict pose
            estimated_pose = model(images.float())

            # compute loss
            loss = compute_loss(estimated_pose, gt, criterion, args)

            epoch_loss += loss.item()
            tepoch.set_postfix(val_loss=loss.item())

    return epoch_loss / len(val_loader)


def train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args):
    epoch_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1

    with tqdm(train_loader, unit="batch") as tepoch:
        for images, gt in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            # for batch_idx, (images, odom) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

            # predict pose
            estimated_pose = model(images.float())

            # compute loss
            loss = compute_loss(estimated_pose, gt, criterion, args)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

            # log tensorboard
            tensorboard_writer.add_scalar('training_loss', loss.item(), iter)

            iter += 1
    return epoch_loss / len(train_loader)  
  

def train(model, train_loader, val_loader, criterion, optimizer, tensorboard_writer, args):
    checkpoint_path = args["checkpoint_path"]
    epochs = args["epoch"]
    best_val = args["best_val"]
    train_losses, val_losses = [], []
    epochs_no_improve = 0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    # 최소 학습률 (0으로 해도 됨)
    for epoch in range(args["epoch_init"], epochs):
        # training for one epoch
        model.train()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args)
        # 스케줄러 업데이트 (매 epoch 끝에 한 번씩 step)
        scheduler.step()

        # 현재 학습률 TensorBoard에 기록 (디버깅용, 선택사항)
        current_lr = optimizer.param_groups[0]['lr']
        tensorboard_writer.add_scalar("learning_rate", current_lr, epoch)
        # validate model
        if val_loader:
            with torch.no_grad():
                model.eval()
                val_loss = val_epoch(model, val_loader, criterion, args)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch: {epoch} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} \n")

            # save best mode
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "best_val": best_val,
                "scheduler_state_dict": scheduler.state_dict(),
            }
            if val_loss < best_val:
                print(f"New best! {best_val:.6f} → {val_loss:.6f}")
                best_val = val_loss
                epochs_no_improve = 0

                state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "best_val": best_val,
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(state, os.path.join(checkpoint_path, "checkpoint_best.pth"))
                torch.save(model.state_dict(), os.path.join(RESULT_DIR, "best_model.pth"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # save checkpoint every 10 epochs
        if not epoch%10:
            torch.save(state, os.path.join(checkpoint_path, "checkpoint_e{}.pth".format(epoch))) 
        # save last checkpoint
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_last.pth"))  

        # log loss in TensorBoard
        tensorboard_writer.add_scalar("train_loss", train_loss, epoch)


def get_optimizer(params, args):
    method = args["optimizer"]

    # initialize the optimizer
    if method == "Adam":
        optimizer = optim.Adam(params, lr=args["lr"])

    elif method == "AdamW":
        optimizer = optim.AdamW(params, lr=args["lr"])

    elif method == "SGD":
        optimizer = optim.SGD(params, lr=args["lr"],
                              momentum=args["momentum"],
                              weight_decay=args["weight_decay"])
    elif method == "RAdam":
        optimizer = optim.RAdam(params, lr=args["lr"])
    elif method == "Adagrad":
        optimizer = optim.Adagrad(params, lr=args["lr"],
                                  weight_decay=args["weight_decay"])

    # load checkpoint
    if args["checkpoint"] is not None:
        checkpoint = torch.load(os.path.join(args["checkpoint_path"], args["checkpoint"]))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer


def compute_loss(y_hat, y, criterion, args):
    if args["weighted_loss"] == None:
        # y_hat, y: (B, 3) 형태 가정 (모델 출력이 요약된 경우)
        loss_main = criterion(y_hat, y.float())

        x_pred, ry_pred, xplus_pred = y_hat[:, 0], y_hat[:, 1], y_hat[:, 2]
        x_gt, ry_gt, xplus_gt = y.float()[:, 0], y.float()[:, 1], y.float()[:, 2]



        loss = (

        )

    else:
        y = torch.reshape(y, (y.shape[0], args["window_size"]-1, 6))
        gt_angles = y[:, :, :3].flatten()
        gt_translation = y[:, :, 3:].flatten()

        # predict pose
        y_hat = torch.reshape(y_hat, (y_hat.shape[0], args["window_size"]-1, 6))
        estimated_angles = y_hat[:, :, :3].flatten()
        estimated_translation = y_hat[:, :, 3:].flatten()

        # compute custom loss
        k = args["weighted_loss"]
        loss_angles = k * criterion(estimated_angles, gt_angles.float())
        loss_translation = criterion(estimated_translation, gt_translation.float())
        loss =  loss_angles + loss_translation   
    return loss


if __name__ == "__main__":

    # set hyperparameters and configuration
    args = {
    "data_dir": "/home/data",
    "bsize": 8,  # batch size
    "optimizer": "AdamW",  # optimizer [Adam, SGD, Adagrad, RAdam]
    "lr": 1e-5,  # learning rate
    "window_size":4,
    "momentum": 0.9,  # SGD momentum
    "weight_decay": 1e-4,  # SGD momentum
    "epoch": 50,  # train iters each timestep
    "weighted_loss": None,  # float to weight angles in loss function
    "pretrained_ViT": "vit_patch16_edim192",  #"vit_patch16_edim384"
    "checkpoint_path": "checkpoints/Exp3",  # path to save checkpoint
    "checkpoint": None,  # checkpoint
    "out_dir": "results",
    "patience": 10,
    "best_val": float('inf'),
    "epoch_init": 0
    }

# tiny  - patch_size=16, embed_dim=192, depth=12, num_heads=3
# small - patch_size=16, embed_dim=384, depth=12, num_heads=6
# base  - patch_size=16, embed_dim=768, depth=12, num_heads=12
    model_params = {
     "dim": 768,
     "image_size": (224, 224),  #"image_size": (384, 384) or (448, 448)
     "patch_size": 16,
     "attention_type": 'divided_space_time',  # ['divided_space_time', 'space_only','joint_space_time', 'time_only']
     "num_frames": args["window_size"],
     "num_classes": 3,  # 6 DoF for each frame
     "depth": 12,
     "heads":6,
     "attn_dropout": 0.1,
     "ff_dropout": 0.2,
     "dropout": 0.6,
     "dim_head": 128,
     "time_only": False,}

    args["model_params"] = model_params

    # create checkpoints folder
    if not os.path.exists(args["checkpoint_path"]):
        os.makedirs(args["checkpoint_path"])

    with open(os.path.join(args["checkpoint_path"], 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args["checkpoint_path"], 'args.txt'), 'w') as f:
    	f.write(json.dumps(args))

    # tensorboard writer
    TensorBoardWriter = SummaryWriter(log_dir=args["checkpoint_path"])

    # train and val dataloader
    print("Using CUDA: ", torch.cuda.is_available())
    print("Loading data...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["bsize"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args["bsize"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )



    # build and load model
    print("Building model...")
    # model_params 일부 적용 (ConvNeXt에 매핑 – patch_size 등은 CNN stem으로 대체)
    model = timm.create_model( #convnext_small.fb_in1k. convnext_base.fb_in1k, convnext_tiny.in12k convnextv2_tiny.fcmae_ft_in1k  convnextv2_small.fcmae_ft_in1k
        'convnext_tiny.fb_in1k',  # ConvNeXt-Tiny (dim≈384, heads-like 구조, depth≈52 blocks but effective 12-like)
        pretrained=True,  # pretrained_ViT처럼 사전 훈련 weights 로드
        num_classes=model_params["num_classes"],  # 3 (VO용 3 DoF – 6 DoF면 6으로)
        drop_rate=model_params["dropout"],  # 0.6
        drop_path_rate=model_params["ff_dropout"],  # 0.2 (ConvNeXt의 stochastic depth ≈ ff_dropout)
    )

    video_model = VideoConvNeXtViT(
        model,
        num_frames=model_params["num_frames"],
        attention_type=model_params["attention_type"],
        attn_dropout=model_params["attn_dropout"],
        dim_head=model_params["dim_head"],
        heads=model_params["heads"],
        time_only=model_params["time_only"]
    )
    video_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # loss and optimizer
    criterion = nn.SmoothL1Loss(beta=0.1)        # Huber Loss 추천 (outlier에 강함)
    optimizer = get_optimizer(video_model.parameters(), args)

    # train network
    print(20*"--" +  " Training " + 20*"--")
    train(video_model, train_loader, val_loader, criterion, optimizer, TensorBoardWriter, args)
