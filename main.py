import numpy as np
import pandas as pd

import torch
from torch import nn

import os
from dataclasses import dataclass
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import balanced_accuracy_score
import argparse
from tqdm import tqdm
import csv

from dataloader import ISICDataset2020
import model

import warnings

from sklearn.metrics import confusion_matrix

DEVICE_IDS = [0]


def build_model(args, backbones, feature_dims):
    if args.strategy == "projection":
        return model.ProjectionEnsemble(backbones, feature_dims=feature_dims)
    elif args.strategy == "soft_vote":
        return model.VotingEnsemble(backbones, mode="soft")
    elif args.strategy == "hard_vote":
        return model.VotingEnsemble(backbones, mode="hard")
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


def get_loaders():
    df = pd.read_csv("/home/gssodhi/isic2020-ensemble/dataset/ISIC_2020_Training_GroundTruth.csv")

    train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df["target"],
            random_state=42,
    )

    root_2020 = '/home/gssodhi/isic2020-ensemble/dataset/train'

    train_dataset = ISICDataset2020(train_df, root_2020, split='train')
    val_dataset = ISICDataset2020(val_df, root_2020, split='val')

    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size = args.batch_size,
            shuffle=True,
            num_workers=args.num_worker,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=True)

    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_worker,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4)

    return train_loader, val_loader, train_df

def run_voting(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, train_df = get_loaders()
    print("=> Loaders Loaded")

    os.makedirs(args.log_file_path, exist_ok=True)  # make sure directory exists

    csv_file = os.path.join(args.log_file_path, f"{args.run}.csv")

    print(f"Loggin in {csv_file}")

    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "f1_score",
                "auc",
                "val_all_probs",
                "train_acc_mel",
                "train_acc_ben",
                "train_precision",
                "train_recall"
            ])

    MODEL_FACTORY = {
        'resnet' : model.ResNet_50_224,
        'effnet' : model.EfficientNet
    }

def train_backbones(args, train_loader, val_loader, train_df, device):
    MODEL_FACTORY = {
        'effnet': model.EfficientNet,
        'resnet': model.ResNet_50_224,
    }
    backbone_names = ['effnet', 'effnet', 'resnet', 'resnet', 'resnet']
    feature_dims = [1792 if name == 'effnet' else 2048 for name in backbone_names]

    labels = np.array(train_df.target)
    class_counts = np.bincount(labels)
    class_weights = [class_counts[1] / sum(class_counts), class_counts[0] / sum(class_counts)]
    pos_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=pos_weights)

    os.makedirs(args.log_file_path, exist_ok=True)
    trained_backbones = []

    for i, name in enumerate(backbone_names):
        print(f"=> Training backbone {i+1}/{len(backbone_names)}: {name}")
        backbone = MODEL_FACTORY[name](in_channels=3, num_classes=2, voting=True).to(device)
        backbone = nn.DataParallel(backbone, device_ids=DEVICE_IDS)
        optimizer = Adam(backbone.parameters(), lr=args.lr)

        csv_file = os.path.join(args.log_file_path, f"backbone_{i}_{name}_{args.run}.csv")
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(args.epochs):
            train_loss, train_acc, _, _, _, _ = train_one_epoch(backbone, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _, _ = test(backbone, val_loader, criterion, device)

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # unwrap DataParallel before stripping the head
        backbone = backbone.module

        # strip classification head so backbone outputs feature vectors
        if name == 'effnet':
            backbone.classifier = nn.Identity()
        else:
            backbone.fc = nn.Identity()

        trained_backbones.append(backbone)

    return trained_backbones, feature_dims


def run_projection(args):
    device = torch.device(f'cuda:{DEVICE_IDS[0]}') if torch.cuda.is_available() else torch.device('cpu')

    ## ==== Data Prep =====
    train_loader, val_loader, train_df = get_loaders()
    print("=> Data Prep isic2020 Done")

    ## ==== Train and freeze backbones =====
    backbones, features = train_backbones(args, train_loader, val_loader, train_df, device)
    print("=> Backbones trained")

    for backbone in backbones:
        for p in backbone.parameters():
            p.requires_grad = False

    # move model to device
    myModel = build_model(args, backbones, features)
    myModel = myModel.to(device)

    labels = np.array(train_df.target)
    class_counts = np.bincount(labels)
    class_weights = [class_counts[1] / sum(class_counts), class_counts[0] / sum(class_counts)]
    pos_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=pos_weights)
    optimizer = Adam(myModel.parameters(), lr=args.lr)

    if torch.cuda.is_available() and len(DEVICE_IDS) > 1:
        print(f"=> Using GPUs {DEVICE_IDS}")
        myModel = nn.DataParallel(myModel, device_ids=DEVICE_IDS)

    os.makedirs(args.log_file_path, exist_ok=True)  # make sure directory exists

    csv_file = os.path.join(args.log_file_path, f"{args.run}.csv")

    print(f"Loggin in {csv_file}")

    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "f1_score",
                "auc",
                "val_all_probs",
                "train_acc_mel",
                "train_acc_ben",
                "train_precision",
                "train_recall"
            ])

    ## ===== Training =====
    for epoch in range(args.epochs):
        train_loss, train_acc, train_acc_mel, train_acc_ben, train_precision, train_recall = train_one_epoch(myModel, train_loader, criterion, optimizer, device)

        save_checkpoint({
                'epoch': epoch,
                'state_dict': myModel.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, filename=args.save_model_path)

        test_loss, test_acc, f1, auc, val_all_probs = test(myModel, val_loader, criterion, device)

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                f1,
                auc,
                val_all_probs,
                train_acc_mel,
                train_acc_ben,
                train_precision,
                train_recall
        ])

        print(f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {test_loss:.4f} | Val Acc: {test_acc:.4f} | "
            f"F1: {f1:.4f} | AUC: {auc:.4f}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0

    all_labels = []
    all_preds = []

    for x, y in tqdm(loader, desc='train'):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        # forward
        out = model(x)

        # loss
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # predictions
        preds = out.argmax(dim=1)

        # Softmax across class dimension
        probs = torch.softmax(out, dim=1)

        all_labels.extend(y.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

        # if 2-class, pick class 1 probabilities
        if probs.shape[1] == 2:
            probs_for_auc = probs[:, 1]
        else:
            probs_for_auc = probs  # multi-class (use full matrix later)

    avg_loss = running_loss / len(loader)

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    acc_ben = cm[0,0] / cm[0].sum()  # TN / N
    acc_mel = cm[1,1] / cm[1].sum()  # TP / P
    TP = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]
    train_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    train_recall    = TP / (TP + FN) if (TP + FN) > 0 else 0

    return avg_loss, balanced_acc, acc_mel, acc_ben, train_precision, train_recall

def test(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc='test'):
            x, y = x.to(device), y.to(device)

            out = model(x)

            loss = criterion(out, y)
            running_loss += loss.item()

            # Softmax across class dimension
            probs = torch.softmax(out, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())


            # for ROC-AUC: if 2-class, pick class 1 probabilities
            if probs.shape[1] == 2:
                probs_for_auc = probs[:, 1]
            else:
                probs_for_auc = probs  # multi-class (use full matrix later)

            all_probs.extend(probs_for_auc.detach().cpu().numpy())

    avg_loss = running_loss / len(loader)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    all_preds = np.array(all_preds)

    num_negatives = np.sum(all_preds == 0)
    num_positive = np.sum(all_preds == 1)

    # compute metrics
    try:
        if len(set(all_labels)) == 2:
            auc = roc_auc_score(all_labels, all_probs)
            f1 = f1_score(all_labels, all_preds, average='binary')
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            f1 = f1_score(all_labels, all_preds, average='macro')
    except ValueError:
        # occurs if only one class present in y
        warnings.warn("WARNING: hit nan in test method")
        auc, f1 = float('nan'), float('nan')

    return avg_loss, balanced_acc, f1, auc, all_probs

def save_checkpoint(state, filename='model'):
    torch.save(state, filename + '.pth.tar')


@dataclass
class model_config:
    batch_size: int = 128
    num_worker:int = 8
    lr: float = 1e-4
    epochs: int = 30
    resume: bool = False
    resume_model_path: str = '/scratch/gssodhi/melanoma/checkpoint/chkpt_efNet'
    save_model_path: str = '/scratch/gssodhi/melanoma/checkpoint/chkpt_efNet'
    log_file_path: str = '/home/gssodhi/melanoma/baselines/data/'
    run: str = 'run'
    freeze: bool = False
    strategy: str = ''


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='RUN Baseline model of ISIC')

    parser.add_argument('--resume',
                        action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--epochs',
                        default=100,
                        type=int)
    parser.add_argument('--strategy',
                        default=None,
                        help='Options: "hard_vote" | "soft_vote" | "projection".',
                        type=str)
    parser.add_argument('--run',
                        type=str)

    cli_args = parser.parse_args()


    run = cli_args.run
    strategy = cli_args.strategy

    args = model_config(
        resume_model_path = f'/home/gssodhi/snap/firmware-updater/224/Desktop/melanoma_detection/ensemble/checkpoint/chkpt_{strategy}_{run}.pth.tar',
        save_model_path = f'/home/gssodhi/snap/firmware-updater/224/Desktop/melanoma_detection/ensemble/checkpoint/chkpt_{strategy}_{run}',
        epochs = cli_args.epochs,
        resume = cli_args.resume,
        batch_size= cli_args.batch_size,
        log_file_path = f'/home/gssodhi/snap/firmware-updater/224/Desktop/melanoma_detection/ensemble/data/{strategy}',
        run = run,
        strategy = cli_args.strategy
    )

    if cli_args.strategy in ['hard_vote', 'soft_vote']:
        run_voting(args)
    elif cli_args.strategy == 'projection':
        run_projection(args)
    else:
        raise ValueError('Strategy dont exist')
