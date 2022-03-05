import argparse
import glob
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from habitat_baselines.rl.ddppo.policy import resnet
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from crl.crl_policy import ResNetEncoder

# class list from:
# https://github.com/yilundu/crl/blob/6adf009d30f292cdc995eb70bab500b0033c11d4/places_finetune/finetune_places.py
INDOOR_CLASSES = [
    "classroom",
    "mansion",
    "patio",
    "airport_terminal",
    "beauty_salon",
    "closet",
    "dorm_room",
    "home_office",
    "bedroom",
    "engine_room",
    "hospital_room",
    "martial_arts_gym",
    "shed",
    "cockpit",
    "hotel-outdoor",
    "apartment_building-outdoor",
    "bookstore",
    "coffee_shop",
    "hotel_room",
    "shopfront",
    "conference_center",
    "shower",
    "conference_room",
    "motel",
    "pulpit",
    "fire_escape",
    "art_gallery",
    "art_studio",
    "corridor",
    "museum-indoor",
    "railroad_track",
    "inn-outdoor",
    "music_studio",
    "attic",
    "nursery",
    "auditorium",
    "residential_neighborhood",
    "cafeteria",
    "office",
    "restaurant",
    "waiting_room",
    "office_building",
    "restaurant_kitchen",
    "stage-indoor",
    "ballroom",
    "game_room",
    "kitchen",
    "restaurant_patio",
    "staircase",
    "banquet_hall",
    "bar",
    "dinette_home",
    "living_room",
    "swimming_pool-outdoor",
    "basement",
    "dining_room",
    "lobby",
    "parlor",
    "locker_room",
]


class PlacessIndoor(Dataset):
    def __init__(self, root):
        self.files, self.labels = [], []
        for idx, c in enumerate(INDOOR_CLASSES):
            folder = os.path.join(root, c)
            if not os.path.isdir(folder):
                print("Warning skipping:", folder)
                continue
            image_files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            self.files.extend(image_files)
            self.labels.extend([idx] * len(image_files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        lbl = self.labels[index]
        return np.array(img), lbl


class PlacesLinear(nn.Module):
    def __init__(self, dim, classes):
        super(PlacesLinear, self).__init__()
        self.fc = nn.Linear(dim, classes)

    def forward(self, x):
        return self.fc(x)


def get_args():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("data", type=str, help="path to dataset")
    parser.add_argument("--ckpt", type=str, help="path to checkpoint")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate (default: 1e-3)")
    parser.add_argument("--epochs", default=60, type=int, help="number of epochs (default: 60)")
    parser.add_argument("-j", "--workers", default=6, type=int, help="number of workers (default: 6)")
    parser.add_argument(
        "-a",
        "--arch",
        default="resnet50",
        type=str,
        choices=["resnet18", "resnet50"],
        help="backbone architecture (default: resnet50)"
    )
    parser.add_argument(
        "-b",
        "--baseplanes",
        default=64,
        type=int,
        choices=[32, 64],
        help="number of baseplanes (default: 64)"
    )
    parser.add_argument(
        "-p",
        "--prefix",
        default="actor_critic.net.ssl_encoder.",
        type=str,
        help="encoder prefix checkpoint (default: actor_critic.net.ssl_encoder.)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="normalize visual inputs (default: false)",
    )
    # fmt: on
    return parser.parse_args()


def train(epoch, train_loader, model, classifier, criterion, optimizer):
    print("==> training...")
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float().cuda()
        target = target.long().cuda()

        input = input.permute(0, 3, 1, 2) / 255.0

        with torch.no_grad():
            feat = model.forward_eval(input)
        output = classifier(feat)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % 10 == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )
            sys.stdout.flush()


def validate(epoch, val_loader, model, classifier, criterion):
    print("==> validtion...")
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float().cuda()
            target = target.long().cuda()

            input = input.permute(0, 3, 1, 2) / 255.0

            feat = model.forward_eval(input)
            output = classifier(feat)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % 10 == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        idx + 1,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )

        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

        with open("metrics.json", "a") as f:
            json.dump(
                {"epoch": epoch + 1, "top1": top1.avg.item(), "top5": top5.avg.item()},
                f,
            )
            f.write("\n")


def main():
    args = get_args()

    # seed
    seed = set_random_seed()
    print(f"==> random seed: {seed}")

    # data
    train_dataset = PlacessIndoor(os.path.join(args.data, "train"))
    print(f"==> training dataset size: {len(train_dataset):,}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        num_workers=args.workers,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = PlacessIndoor(os.path.join(args.data, "val"))
    print(f"==> validation dataset size: {len(val_dataset):,}")
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
    )

    # model
    model = ResNetEncoder(
        observation_space=spaces.Dict({"rgb": spaces.Box(0, 255, (256, 256, 3))}),
        baseplanes=args.baseplanes,
        make_backbone=getattr(resnet, args.arch),
        normalize_visual_inputs=args.normalize,
    ).cuda()
    classifier = PlacesLinear(model.backbone.final_channels, len(INDOOR_CLASSES)).cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().cuda()

    # load checkpoint
    if args.ckpt is not None:
        print(f"==> loading checkpoint: {args.ckpt}")

        checkpoint = torch.load(args.ckpt, map_location="cpu")
        if "extra_state" in checkpoint and "step" in checkpoint["extra_state"]:
            step = checkpoint["extra_state"]["step"]
            print(f"==> checkpoint step: {step:,}")

        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith(args.prefix):
                state_dict[k[len(args.prefix) :]] = state_dict[k]
            del state_dict[k]
        model.load_state_dict(state_dict, strict=True)

    # training
    for epoch in range(args.epochs):
        train(epoch, train_loader, model, classifier, criterion, optimizer)
        validate(epoch, val_loader, model, classifier, criterion)
        if epoch == 0:
            double_check(model, args)


def double_check(model, args):
    if args.ckpt is None:
        return
    print("==> checking that encoder parameters have not changed...")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict_old = checkpoint["state_dict"]
    state_dict_new = model.state_dict()

    for k in list(state_dict_new.keys()):
        assert torch.all(state_dict_new[k].cpu() == state_dict_old[args.prefix + k])
    print("==> check passed")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def set_random_seed():
    # adapted from detectron2
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


if __name__ == "__main__":
    main()