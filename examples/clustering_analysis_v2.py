import os
from pyexpat import features
import torch
import cv2
import numpy as np
import glob
import json
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from habitat_baselines.rl.ddppo.policy import resnet_gn
from PIL import ImageColor, Image
from torch.utils.data import DataLoader, Dataset

colors = [
    '#4f7ac9',
    '#e68752',
    '#6fc769',
    '#d06565',
    '#9470b0',
    '#886140',
    '#d783be',
    '#797979',
    '#d0b86d',
    '#87c4dd',
    '#4f7ac9',
    '#e68752',
    '#6fc769',
    '#d06565',
    '#9470b0',
    '#125688',
    '#4dc247',
    '#cb2027',
    '#00bf8f',
    '#fffc00',
    '#f4a5a5',
    ]
TRAJECTORY="trajectory_4"

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

INDOOR_CLASSES = [
    "closet",
   "home_office",
   "bedroom",
   "hotel_room",
   "shower",
   "corridor",
   "kitchen",
   "staircase",
   "dinette_home",
   "living_room",
   "dining_room",
   "lobby",
   "parlor",
]

class PlacessIndoor(Dataset):
    def __init__(self, root):
        self.files, self.labels = [], []
        #classes = random.sample(INDOOR_CLASSES, 10)
        for idx, c in enumerate(INDOOR_CLASSES):
            folder = os.path.join(root, c)
            if not os.path.isdir(folder):
                print("Warning skipping:", folder)
                continue
            image_files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            self.files.extend(image_files)
            self.labels.extend([c] * len(image_files))
        print("Samples: {}".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        lbl = self.labels[index]
        return np.array(img), lbl

def plot_tsne(model, device, images, targets, ckpt):
    model.eval()
    features = []; labels = []
    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(tqdm(zip(images, targets))):
            data = torch.tensor(image).unsqueeze(0).float().to(device)
            # print(data.shape, target.shape)
            # target = target[0]]), axis=0)

            pred = model(data / 255.0)
            pred = model.avgpool(pred).squeeze(-1).squeeze(-1).detach().cpu().numpy()
            if batch_idx == 0: 
                features = pred
                labels = target
            else: 
                features = np.concatenate((features, pred), axis=0)
                labels = np.concatenate((labels, target), axis=0)
    print("before pca: {}".format(features.shape))
    pca = PCA(n_components=30)
    pca_result = pca.fit_transform(features)
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=500)
    tsne_results = tsne.fit_transform(pca_result)
    
    tx = tsne_results[:, 0]
    ty = tsne_results[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    rooms = np.unique(labels).tolist()
    color_map = {}
    for i in range(len(rooms)):
        color_map[rooms[i]] = colors[i]

    visualize_tsne_points(tx, ty, labels, ckpt, color_map)
    visualize_tsne_images(tx, ty, labels, images, ckpt, color_map)
    # visualize_class_concepts_tsne(features, labels, ckpt, color_map)

    return tsne_results

def plot_places_tsne(model, device, dataloader, ckpt):
    model.eval()
    features = []; labels = []
    images = []
    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(tqdm(dataloader)):
            data = torch.tensor(image).permute(0, 3, 1, 2).float().to(device)
            # print(data.shape, target.shape)
            # target = target[0]]), axis=0)

            pred = model(data / 255.0)
            pred = model.avgpool(pred).squeeze(-1).squeeze(-1).detach().cpu().numpy()
            # print(pred.shape)
            if batch_idx == 0: 
                features = pred
                labels = target
                images = image
            else: 
                images = np.concatenate((images, image), axis=0)
                features = np.concatenate((features, pred), axis=0)
                labels = np.concatenate((labels, target), axis=0)
    pca = PCA(n_components=30)
    pca_result = pca.fit_transform(features)
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=500)
    tsne_results = tsne.fit_transform(pca_result)
    
    tx = tsne_results[:, 0]
    ty = tsne_results[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    rooms = np.unique(labels).tolist()
    # colors = sns.color_palette(n_colors=len(rooms))
    color_map = {}
    for i in range(len(rooms)):
        color_map[rooms[i]] = colors[i]

    visualize_tsne_points(tx, ty, labels, ckpt, color_map, plot_id="places")
    # visualize_class_concepts_tsne(features, labels, ckpt, color_map, plot_id="places")
    visualize_tsne_images(tx, ty, labels, images, ckpt, color_map, plot_id="places")

    return tsne_results

def visualize_tsne_points(tx, ty, labels, ckpt, color_map, plot_id="resnet50_tsne_v2"):
    df = pd.DataFrame({"tsne-2d-one": tx, 'tsne-2d-two': ty, 'labels': labels})

    pallete_size = np.unique(labels).shape[0]

    print(color_map)

    plt.figure(figsize=(16,10))
    sns.scatterplot(data=df, x="tsne-2d-one", y="tsne-2d-two", hue="labels", legend="full")
    plt.savefig('demos_v2/{}/demos/{}_{}.pdf'.format(TRAJECTORY, plot_id, ckpt), bbox_inches='tight', dpi=300)

def visualize_class_concepts_tsne(features, labels, ckpt, color_map, plot_id="resnet50_tsne_v2"):
    categories = np.unique(labels)
    avg_features = []
    avg_labels = []
    for i, category in enumerate(categories.tolist()):
        avg_feats = features[labels == category]
        avg_feats = np.expand_dims(np.mean(avg_feats, axis=0), axis=0)
        #print("feats", avg_feats.shape)
        if i == 0:
            avg_features = avg_feats
            avg_labels = [category] 
        else:
            avg_features = np.concatenate((avg_features, avg_feats), axis=0)
            avg_labels = np.concatenate((avg_labels, [category]), axis=0)
    print("avg feats: {} {}".format(avg_features.shape, avg_labels.shape))

    pca_2 = PCA(n_components=20)
    # pca_result = pca_2.fit_transform(np.zeros((30, 100)))
    print("avg", avg_features.shape)
    pca_result = pca_2.fit_transform(avg_features)
    print(pca_result.shape, avg_features.shape)
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=500)
    tsne_results = tsne.fit_transform(pca_result)

    tx = tsne_results[:, 0]
    ty = tsne_results[:, 1]
    print(tx.shape, avg_labels.shape)
    df = pd.DataFrame({"tsne-2d-one": tx, 'tsne-2d-two': ty, 'labels': avg_labels})

    pallete_size = np.unique(labels).shape[0]

    print(color_map)

    plt.figure(figsize=(16,10))
    ax = sns.scatterplot(data=df, x="tsne-2d-one", y="tsne-2d-two", hue="labels", \
        palette=color_map, legend=False)
    
    for i, txt in enumerate(avg_labels):
        print(txt)
        ax.annotate(txt, (tx[i], ty[i]))
    plt.savefig('demos_v2/{}/demos/{}_concepts_v2_{}.pdf'.format(TRAJECTORY, plot_id, ckpt), bbox_inches='tight', dpi=300)


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset
 
    # knowing the image center,
    # compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)
 
    br_x = tl_x + image_width
    br_y = tl_y + image_height
 
    return tl_x, tl_y, br_x, br_y

def visualize_tsne_images(tx, ty, labels, images, ckpt, color_map, plot_size=2000, max_image_size=100, plot_id="resnet50_tsne_v2"):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot

    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    # init the plot as white canvas
    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)
    count = 0

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    sampled_idx = random.sample(np.arange(0, images.shape[0]).tolist(), 100)
    for image, label, x, y in tqdm(
            zip(images[sampled_idx], labels[sampled_idx], tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        count += 1
        # if count%10 != 0:
        #     continue
        # image = np.transpose(image, (1, 2, 0))

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)
    
        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label, color_map)
    
        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)
    
        # put the image to its t-SNE coordinates using numpy sub-array indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    count = 0
    for room, color in color_map.items():
        count += 1
        color = ImageColor.getcolor(color, "RGB")
        cv2.circle(tsne_plot, (20, 60*count), 12, color, -1)
        cv2.putText(tsne_plot, room, (40, 10+60*count), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, color, 3, cv2.LINE_AA)

    # cv2.imshow('t-SNE', tsne_plot)
    cv2.imwrite('demos_v2/{}/demos/{}_{}.jpg'.format(TRAJECTORY, plot_id, ckpt), tsne_plot)

    # cv2.waitKey()

def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image

def scale_to_01_range(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def draw_rectangle_by_class(image, label, color_map):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = color_map[label]
    color = ImageColor.getcolor(color, "RGB")
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=8)

    return image

def get_model(model_path=None):
    model = resnet_gn.resnet50(3, 32, 16)

    state_dict = torch.load(model_path, map_location="cpu")["teacher"] 
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    
    if model_path is not None:
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model

def get_finetuned_model(model_path):
    model = resnet_gn.resnet50(3, 32, 16)

    state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if "actor_critic.net.visual_encoder.backbone" in k}
    state_dict = {k.replace("actor_critic.net.visual_encoder.backbone.", ""): v for k, v in state_dict.items()}

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    return model

def load_trajectory():
    images = []
    image_paths = []

    rooms = []

    room_map = {}
    files = os.listdir("demos_v2/{}/demos/".format(TRAJECTORY))
    # files = ["1LXtFkjw3qL_{}".format(i) for i in range(0, 39)]
    room_image_map = {}
    for f in tqdm(files):
        if f.endswith(".mp4"):
            continue
        meta_path = "demos_v2/{}/demos/{}/meta.json".format(TRAJECTORY, f)
        traj_path = "demos_v2/{}/demos/{}".format(TRAJECTORY, f)
        if not os.path.exists(meta_path):
            continue
        demo_id = f
        scene_id = demo_id.split("_")[0]
        with open(meta_path, "r") as f:
            meta = json.load(f)
            for i, step in enumerate(meta):
                if i%5 != 0:
                    continue
                filename = step["frame_id"] # "{}_{}.png".format(demo_id, i)        
                img_path = os.path.join(traj_path, filename)
                img = cv2.imread(img_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))
                img = np.transpose(img, (2, 0, 1))

                if len(step["room"]) > 0 and step["room"][0].split("/")[0] not in ["hallway", "stairs", "workout", "stairs", "other room", "entryway", "office", "porch", "spa", "balcony"]:
                    room = step["room"][0].split("/")[0]
                    if scene_id not in room_image_map.keys():
                        room_image_map[scene_id] = {}
                    if room not in room_image_map[scene_id].keys():
                        room_image_map[scene_id][room] = []
                    room_image_map[scene_id][room].append(img)
                    if room_map.get(room) is None:
                        room_map[room] = 1
                    rooms.append([room])
                    images.append(img)
                    image_paths.append(img_path)
    images = []
    rooms = []
    for scene, room_img_mp in room_image_map.items():
        for room, imgs in room_img_mp.items():
            print(room, len(imgs))
            rooms.extend([[room] for i in range(min(len(imgs), 20))])
            images.extend(random.sample(imgs, min(len(imgs), 20)))

    print("Total frames: {}, {}".format(len(images), len(rooms)))
    print("Total categories: {}".format(room_map.keys()))
    return np.array(images), np.array(rooms), image_paths, room_map


def main():
    images, targets, image_paths, room_map = load_trajectory()

    model_path = "data/new_checkpoints/rgb_encoders/omnidata_DINO_02.pth"
    pretrained_model = get_model(model_path)

    finetuned_model_path = "data/new_checkpoints/rgb_encoders/ckpt.99.pth"
    finetuned_model = get_finetuned_model(finetuned_model_path)

    scratch_model_path = "data/new_checkpoints/rgb_encoders/ckpt_scratch.99.pth"
    scratch_model = get_finetuned_model(scratch_model_path)

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    pretrained_model.to(device)
    scratch_model.to(device)
    finetuned_model.to(device)

    data_path = "/srv/flash1/skuhar6/scaling_crl/new_scaling/temp_scaling/data/places365_standard/"
    
    if False:
        plot_tsne(pretrained_model, device, images, targets, "pretrained")
        plot_tsne(scratch_model, device, images, targets, "scratch")
        plot_tsne(finetuned_model, device, images, targets, "finetuned")
    else:
        val_dataset = PlacessIndoor(os.path.join(data_path, "val"))
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            drop_last=False,
        )
        plot_places_tsne(pretrained_model, device, val_loader, "pretrained")
        plot_places_tsne(scratch_model, device, val_loader, "scratch")
        plot_places_tsne(finetuned_model, device, val_loader, "finetuned")


if __name__ == '__main__':
    main()

