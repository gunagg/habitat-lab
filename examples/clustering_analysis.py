from cgitb import text
import os
import torch
import cv2
import numpy as np
import imageio
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from habitat_baselines.rl.ddppo.policy import resnet_gn

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
    pca = PCA(n_components=30)
    pca_result = pca.fit_transform(features)
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=500)
    tsne_results = tsne.fit_transform(features)
    df = pd.DataFrame({"tsne-2d-one": tsne_results[:, 0], 'tsne-2d-two': tsne_results[:, 1], \
        "labels": labels})

    pallete_size = np.unique(labels).shape[0]

    colors = ['#4f7ac9', '#e68752', '#6fc769', '#d06565', '#9470b0', '#886140', '#d783be', '#797979', '#d0b86d', '#87c4dd', '#4f7ac9', '#e68752', '#6fc769', '#d06565', '#9470b0']
    rooms = np.unique(labels).tolist()
    color_map = {}
    for i in range(len(rooms)):
        color_map[rooms[i]] = colors[i]

    print(color_map)

    plt.figure(figsize=(16,10))
    sns.scatterplot(data=df, x="tsne-2d-one", y="tsne-2d-two", hue="labels", \
        palette=color_map, legend="full")
    plt.savefig('demos/resnet50_tsne_v2_{}.pdf'.format(ckpt), bbox_inches='tight', dpi=300)
    return tsne_results

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
    image_number = []

    rooms = []

    room_map = {}
    files = ["1LXtFkjw3qL_{}".format(i) for i in range(0, 39)]

    for f in files:
        meta_path = "demos/{}/meta.json".format(f)
        traj_path = "demos/{}".format(f)
        demo_id = f
        with open(meta_path, "r") as f:
            meta = json.load(f)
            for i, step in enumerate(meta):
                filename = "{}_{}.png".format(demo_id, i)        
                image_number.append(int(filename.split(".")[0].split("_")[-1]))
                img_path = os.path.join(traj_path, filename)
                img = cv2.imread(img_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))
                img = np.transpose(img, (2, 0, 1))

                if len(step["room"]) > 0 and step["room"][0] != "hallway":
                    room = step["room"][0].split("/")[0]
                    if room_map.get(room) is None:
                        room_map[room] = 1
                    rooms.append([room])
                    images.append(img)
    print("Total frames: {}".format(len(images)))
    print("Total categories: {}".format(len(room_map.keys())))
    # images = [x for _, x in sorted(zip(image_number, images))]
    return np.array(images), np.array(rooms), room_map


def main():
    model_path = "data/new_checkpoints/rgb_encoders/omnidata_DINO_02.pth"
    model_pretrained = get_model(model_path)

    scratch_path = "data/new_checkpoints/rgb_encoders/ckpt_scratch.99.pth"
    model_scratch = get_finetuned_model(scratch_path)


    scratch_path = "data/new_checkpoints/rgb_encoders/ckpt.99.pth"
    model_finetuned = get_finetuned_model(scratch_path)

    images, targets, room_map = load_trajectory()

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    model_pretrained.to(device)
    model_scratch.to(device)
    model_finetuned.to(device)

    plot_tsne(model_pretrained, device, images, targets, "pretrained")
    plot_tsne(model_scratch, device, images, targets, "scratch")
    plot_tsne(model_finetuned, device, images, targets, "finetuned")


if __name__ == '__main__':
    main()