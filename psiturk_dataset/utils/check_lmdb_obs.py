import argparse
import lmdb
import torch
import msgpack_numpy
import numpy as np

from habitat.utils.visualizations.utils import observations_to_image, images_to_video

def read_observation_from_lmdb(lmdb_env, lmdb_txn, lmdb_cursor, idx=0):
    obs_idx = "{0:0=6d}_obs".format(idx)
    observations_binary = lmdb_cursor.get(obs_idx.encode())
    observations = msgpack_numpy.unpackb(observations_binary, raw=False)
    for k, v in observations.items():
        obs = np.array(observations[k])
        observations[k] = torch.from_numpy(obs)
    
    obs_list = []
    print("Observations shape: {}".format(observations["rgb"].shape))
    for i in range(observations["rgb"].shape[0]):
        frame = observations_to_image({"rgb": observations["rgb"][i]}, {})
        obs_list.append(frame)
    
    images_to_video(obs_list, output_dir="demos", video_name="re_create_{}".format(idx))


def get_videos_from_lmdb(path, ub):
    lmdb_env = lmdb.open(
            path,
            map_size=int(1e10),
            writemap=True,
        )
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    for i in range(0, ub):
        read_observation_from_lmdb(lmdb_env, lmdb_txn, lmdb_cursor, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/datasets/object_rearrangement/v0/train/train.db"
    )
    parser.add_argument(
        "--ub", type=int, default=10
    )
    args = parser.parse_args()

    get_videos_from_lmdb(args.path, args.ub)