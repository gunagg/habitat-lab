import argparse
import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import gzip

from collections import defaultdict
from psiturk_dataset.utils.utils import write_json, write_gzip, load_dataset, load_json_dataset
from tqdm import tqdm

# eval success episodes
# episode_ids = ['A1L3937MY09J3I:3Z7EFSHGNBH1CG7U84ECI5ABGYYCX5','A1ZE52NWZPN85P:3C6FJU71TSWMYFE4ZRLEVP3QQU9YUY','A2CWA5VQZ6IWMQ:3YGXWBAF72KAEEJKOTC7LUDDNIP4C8','APGX2WZ59OWDN:358010RM5GWXBPDUZL9H8XY015IVXR']
# train success episodes
# episode_ids = ['A1NKBXOTZAI1YK:3SEPORI8WP22OWABP8669V0YPHWAZS','A1ZE52NWZPN85P:39OWYR0EPMUXFXHE42QF9P2NG19YF2','A272X64FOZFYLB:30X31N5D65T5NKOXUGCYD23V2FUSAT','A2CWA5VQZ6IWMQ:3OS46CRSLH2KSATYYY0R8KLG5LU6VO','A2Q6L9LKSNU7EB:323Q6SJS8KJBT2RPU2MRNP7KQ5NHFD','A2Q6L9LKSNU7EB:39U1BHVTDNU6IZ2RA12E0ZLBY95T3I']
# resnet1 eps
episode_ids = ['AKYXQY5IP7S0Z:3I0BTBYZAZO6IT2O1K7U6IFJAEIY0Q','A1NSHNH3MNFRGW:3GLB5JMZFZY0VMIIJQ9JEPSYZ2JGDB','A2TUUIV61CR0C7:3HYA4D452TM7ECO7BHJK0L1I02B2FH','AEWGY34WUIA32:3ZY8KE4ISL6D2SCID7EPEP274HEQV2','A1NSHNH3MNFRGW:3FTF2T8WLTLKPIV1MF8ZEWVW2V4W9Z','A272X64FOZFYLB:36W0OB37HYHHYJIPVEGYQHN22FPZH1','A1ZE52NWZPN85P:3VZLGYJEYNDEK9I40IYKT3BWQ0SZX4','ANBWJZYU2A68T:3LKC68YZ3C6NW5Z7O4RHBMQLWPTOWX','ADEMAGRRRGSPT:39JEC7537W498R2Z8PDUUKDQZAWVC0','AKYXQY5IP7S0Z:3ZQIG0FLQGJIMP84PGDV6EKTTXIWVU','A272X64FOZFYLB:3R3YRB5GRH6L2XG1JL7YS3LJNX7AUP','AKYXQY5IP7S0Z:3YJ6NA41JDJJBLB9W5LHBW13593PJU','A1NSHNH3MNFRGW:32M8BPYGAVPH3XY4B4AU5M8BRXWGIY','A272X64FOZFYLB:3X73LLYYQ3HNHU46SQ54VUGTSW9HNK','A182N7RLXGSCZG:3KGTPGBS6ZOWXULX66EJML2LAPC2U2','A272X64FOZFYLB:3VA45EW49PQUV4J4RG2WIW0R1HN1OP','A3O5RKGH6VB19C:320DUZ38G9PDY8IATMVUHNNB2FEGJ7','A2CWA5VQZ6IWMQ:3NLZY2D53RSA6N0OZ3CJRG45EPLLQ0','A2TUUIV61CR0C7:33TIN5LC06DOENQ11GQNZTGCDJHY99','AKYXQY5IP7S0Z:3Z4XG4ZF4AUZ0DHHRSY7GJESR1HX8D','A1ZE52NWZPN85P:32ZKVD547HQ6MD8AAFBT05FPS4GB3P','AEWGY34WUIA32:34S9DKFK75S93PUV2Q9SHUBWHP4YNB','A1ZE52NWZPN85P:36PW28KO41Z4D1JFTLSTOLZG1P4EAZ','A3O5RKGH6VB19C:3MMN5BL1W17254C71412ELQJ4663MY','A272X64FOZFYLB:3WS1NTTKE0F0I2LTWUF6HX834OLF09','AKYXQY5IP7S0Z:3483FV8BEGMBVJVWAOGG6FO573B26H','A3O5RKGH6VB19C:3EKVH9QME07AGSABKBOUCLYXOSK2D1','A1NKBXOTZAI1YK:3NG53N1RLXMUR4FQ51OQM6SPP8ZP8K','A272X64FOZFYLB:3TYCR1GOTEMJKF1FMZVWI9G9JRWLZI','A272X64FOZFYLB:324G5B4FB5BN396NEBHUT5VM60X07L','A2CWA5VQZ6IWMQ:3RGU30DZTCBDQIEW4PTPUS780Y3MJU','ADEMAGRRRGSPT:3WEV0KO0OOV3LRR9EQ3033B1NDESD8','ADEMAGRRRGSPT:3KRVW3HTZPO6PLXMRJ23MTYVVCPMS8','A2TUUIV61CR0C7:3IAEQB9FMGNWS88IYVD10SEMTWWWDP','A2CWA5VQZ6IWMQ:3EF8EXOTT3YGUTS7B3ARA0J52TKJ17','A272X64FOZFYLB:3L4D84MIL1VRY4DLDSDC2NZCJC4HJG','AOMFEAWQHU3D8:32XVDSJFP10DKMGOX4NXVBLRYXF2M5','APGX2WZ59OWDN:39JEC7537W498R2Z8PDUUKDQ02SCVY','A1L3VZLK8DCFNM:31IBVUNM9U2GB3M9ZR3V2QYTWO1VFQ','A2TUUIV61CR0C7:3AQF3RZ55ALVWD78YJVNQYIUH166FK','A2DDPSXH2X96RF:37C0GNLMHH6YYTTC7D0X2YF95OF6DY','AEWGY34WUIA32:32XVDSJFP10DKMGOX4NXVBLRXJU2MR','A2Q6L9LKSNU7EB:3BEFOD78W8WNN0VB1I6LOQIPIB9M4J','A2CWA5VQZ6IWMQ:333U7HK6IBIAMO8JRWUMB2KERKHJDF','AEWGY34WUIA32:3UN61F00HYSWGZC3KVLCFHIDOU4R5I','AOMFEAWQHU3D8:34PGFRQONQE9VU8A8RZC3Q9ZY33WJN','A3O5RKGH6VB19C:3FIUS151DX5376S9LGARKAVVANYGG4','A3KC26Z78FBOJT:369J354OFFD1AD3393158JI6IGVG6I','A1ZE52NWZPN85P:3D8YOU6S9GNKFV4YT8QMCYJXRGM6U4','A1NSHNH3MNFRGW:3FPRZHYEP0ALVR6GFW2T1H9WUGUV3G','A1NSHNH3MNFRGW:33LK57MYLV86OSW568SXUVU4BLXSZW','A1NSHNH3MNFRGW:386PBUZZXH0TK0WB4DSAUFSJ034LJ2','A1NSHNH3MNFRGW:32ZKVD547HQ6MD8AAFBT05FPS2CB3H','A2TUUIV61CR0C7:3R2PKQ87NYBHV7UQM78PIRS8NTDIMX','A1ZE52NWZPN85P:3KKG4CDWKK18GGCHC92GJ4C5I8T49L','A3TYITGZ7LBO54:3EG49X351WFCWZYTYD19W5I1LWW6XL','AOMFEAWQHU3D8:3TK8OJTYM3OS2GB3DUZ0EKCX00IVP1','A272X64FOZFYLB:358010RM5GWXBPDUZL9H8XY02P0XVG','A1ZE52NWZPN85P:38JBBYETQQDPBC3YKKI2BIDG91M4E0','A1NSHNH3MNFRGW:3PJUZCGDJ8J9ZHZJOCST0GSAKUI98Q','A3O5RKGH6VB19C:37XITHEISYCHFKLIZ58KTNONFKMCRO','A1NSHNH3MNFRGW:34MAJL3QP6QM1EN1V016SR9JIAQ43Z','A1NSHNH3MNFRGW:3V5Q80FXIZUCY08ERMIIZCCLYWY235','A2Q6L9LKSNU7EB:3OB0CAO74JSHTT8KZSEFCAE0WQ9YH7','A3O5RKGH6VB19C:3RUIQRXJBDRZFQKB7Y4NAU5B2L4LLJ','A2TUUIV61CR0C7:3MHW492WW2GMHDEQLE78XGI2VV0VMY','A2CWA5VQZ6IWMQ:3C6FJU71TSWMYFE4ZRLEVP3QQU4UYP','APGX2WZ59OWDN:39OWYR0EPMUXFXHE42QF9P2NFWAYFS','A272X64FOZFYLB:33JKGHPFYEX9985HJNLHNZOP9I3NMU','A3K1P4SPR8XS7R:3P1L2B7AD3S7LBN8KQKF2B95XQOLOF']


def load_duplicate_episodes(path="data/hit_data/duplicate_episode.json"):
    f = open(path, "r")
    data = json.loads(f.read())
    episode_ids = [d["episode_id"] for d in data]
    return episode_ids


def sample_episodes(path, output_path, per_scene_limit=10):
    data = load_dataset(path)

    print("Number of episodes {}".format(len(data["episodes"])))

    sample_episodes = {}
    sample_episodes["instruction_vocab"] = data["instruction_vocab"]
    sample_episodes["episodes"] = []
    scene_map = {}
    for episode in data["episodes"]:
        scene_id = episode["scene_id"]

        if scene_id not in scene_map.keys():
            scene_map[scene_id] = 0
        scene_map[scene_id] += 1 

        if scene_map[scene_id] <= per_scene_limit:
            sample_episodes["episodes"].append(episode)
    
    print("Sampled episodes: {}".format(len(sample_episodes["episodes"])))
    
    write_json(sample_episodes, output_path)
    write_gzip(output_path, output_path)


def sample_episodes_by_episode_ids(path, output_path):
    episode_file = open(path, "r")
    data = json.loads(episode_file.read())

    episode_ids = load_duplicate_episodes()

    print("Number of episodes {}".format(len(data["episodes"])))
    print("Number of duplicate episodes {}".format(len(episode_ids)))
    print("Sampling {} episodes".format(len(episode_ids)))

    sample_episodes = {}
    sample_episodes["instruction_vocab"] = data["instruction_vocab"]
    sample_episodes["episodes"] = []
    scene_map = {}
    for episode in data["episodes"]:
        scene_id = episode["scene_id"]
        episode_id = episode["episode_id"]
        # Exclude episodes
        if episode_id not in episode_ids:
            sample_episodes["episodes"].append(episode)
    
    print("Sampled episodes: {}".format(len(sample_episodes["episodes"])))
    
    write_json(sample_episodes, output_path)
    write_gzip(output_path, output_path)


def sample_objectnav_episodes(path, output_path, prev_tasks):
    prev_tasks = ["data/datasets/objectnav_mp3d_v2/train/sampled", "data/datasets/objectnav_mp3d_v2/train/sampled_v2"]
    prev_episode_points = {}
    for prev_path in prev_tasks:
        prev_task_files = glob.glob(prev_path + "/*.json")
        for prev_task in prev_task_files:
            data = load_json_dataset(prev_task)
            for ep in data["episodes"]:
                key = str(ep["start_position"])
                if key not in prev_episode_points.keys():
                    prev_episode_points[key] = 0
                prev_episode_points[key] += 1

    files = glob.glob(path + "/*.json.gz")
    hits = []
    scene_ep_map = defaultdict(int)
    num_duplicates = 0
    total_episodes = 0
    print("Number of existing episodes: {}".format(len(prev_episode_points.keys())))
    for file_path in files:
        data = load_dataset(file_path)
        scene_id = file_path.split("/")[-1].split(".")[0]
        object_category_map = defaultdict(int)
        episodes = []
        count = 0
        for episode in data["episodes"]:
            key = str(episode["start_position"])
            if prev_episode_points.get(key) is not None and prev_episode_points[key] > 0:
                num_duplicates += 1
                continue
            object_category = episode["object_category"]
            if object_category_map[object_category] < 15:
                object_category_map[object_category] += 1
                episodes.append(episode)
                if key not in prev_episode_points.keys():
                    prev_episode_points[key] = 0
                prev_episode_points[key] += 1
                count += 1

        data["episodes"] = episodes
        dest_path = os.path.join(output_path, "{}.json".format(scene_id))
        # print(output_path)
        # print(dest_path)
        write_json(data, dest_path)
        write_gzip(dest_path, dest_path)

        scene_ep_map[scene_id] = len(data['episodes'])

        # data["episodes"] = episodes[:1]
        # dest_path = os.path.join(output_path, "{}_train.json".format(scene_id))
        # write_json(data, dest_path)
        total_episodes += len(episodes)

        ep = {
            "name": "{}.json".format(scene_id),
            "config": "tasks/objectnav_v2/{}.json".format(scene_id),
            "scene": "{}.glb".format(scene_id),
            "trainingTask": {
                "name": "{}_train.json".format(scene_id),
                "config": "tasks/objectnav_v2/{}_train.json".format(scene_id)
            }
        }

        hits.append(ep)
    # print(json.dumps(hits, indent=4))
    with open("hits.json", "w") as f:
        f.write(json.dumps(hits, indent=4))
    with open("scene_ep_map.json", "w") as f:
        f.write(json.dumps(scene_ep_map))

    print("Number of new episodes: {}".format(total_episodes))
    print("Number of duplicate episodes: {}".format(num_duplicates))
    print("Number of new episodes: {}".format(len(prev_episode_points.keys())))


def sample_episodes_by_scene(path, output_path, limit=500):
    data = load_dataset(path)

    ep_inst_map = {}
    episodes = []
    excluded_episodes = []
    for ep in tqdm(data["episodes"]):
        instruction = ep["instruction"]["instruction_text"].replace(" ", "_")
        scene_id = ep["scene_id"]

        if scene_id not in ep_inst_map.keys():
            ep_inst_map[scene_id] = {}
        else:
            if instruction not in ep_inst_map[scene_id].keys():
                ep_inst_map[scene_id][instruction] = 1
                episodes.append(ep)
            else:
                ep_inst_map[scene_id][instruction] += 1
                excluded_episodes.append(ep)

    sample_length = limit - len(episodes)
    if sample_length > 0:
        sampled = np.random.choice(excluded_episodes, sample_length)
        episodes.extend(sampled.tolist())
    else:
        sampled = np.random.choice(episodes, limit)
        episodes = sampled.tolist()

    data["episodes"] = episodes

    write_json(data, output_path)
    write_gzip(output_path, output_path)


def sample_objectnav_episodes_custom(path, output_path):
    files = glob.glob(path + "/*.json.gz")
    print("In ObjectNav episode sampler")
    # episode_ids = ["A2DDPSXH2X96RF:3DPNQGW4LNILYXAJE2Z4ZUL33AU642", "A7XL1V3G7C2VV:3QY5DC2MXTNGYOX9U1TQ64WAVJ6UFV", "A272X64FOZFYLB:3SNLUL3WO6Q2YG75GCWO1H1UQ67LUM", "AEWGY34WUIA32:32UTUBMZ7IZQYMATUPHZJ078SJ2BVE", "AOMFEAWQHU3D8:34S9DKFK75S93PUV2Q9SHUBWT7RYNA", "AOMFEAWQHU3D8:3MRNMEIQW79GHEWJUH6ZRHX66OODLI", "A1ZE52NWZPN85P:3TOK3KHVJVL86QY6GWJ5J6R4F8I7O8", "A3KC26Z78FBOJT:3NC5L260MQPLLJDCYFHH7Y4LD55FO6"]
    # train episodes
    episode_ids = ["A1ZE52NWZPN85P:36U2A8VAG328VJ9S5DHCP2USZ2CKYX", "A2TUUIV61CR0C7:3DI28L7YXCH8JD6FX2Z0DK6D7D91EJ", "A1R0689JPSQ3OF:354GIDR5ZD99LY63TCWLEQLZ9RU009", "A1ZE52NWZPN85P:3TOK3KHVJVL86QY6GWJ5J6R4F8I7O8", "A3PFU4042GIQLE:3EFE17QCRE8KX7WB0MMQUOQZQA2HSM", "A29CNQYJWMJ7G9:33TIN5LC06DOENQ11GQNZTGCOHVY9U", "ADEMAGRRRGSPT:3TXWC2NHN1TRI1ES2AYYH7SB732S9R", "AOMFEAWQHU3D8:3MRNMEIQW79GHEWJUH6ZRHX66OODLI", "A3KC26Z78FBOJT:3CPLWGV3MQ2U2OMNUEHCIDI5KOH9NN", "A1ZE52NWZPN85P:3FE2ERCCZZBXCW26CIDMJSIPF6VOPB", "A29CNQYJWMJ7G9:3GU1KF0O4K4DT2DX8D80D8IQ8CWBPO", "A2TUUIV61CR0C7:386PBUZZXH0TK0WB4DSAUFSJB8XLJG", "A2TUUIV61CR0C7:3U5JL4WY5MCYHCUFFP8UZ7YNZ4MX4V", "A272X64FOZFYLB:3SNLUL3WO6Q2YG75GCWO1H1UQ67LUM", "A1NSHNH3MNFRGW:3QAVNHZ3EO7IJ7T7A7FX1GP0K52ALR", "A3KC26Z78FBOJT:30OG32W0SWEBXKD42PXYARJGBPCENP", "A3C7COPV48I37D:378XPAWRUEGGT6L1P4IK90X84NLIA7", "A2Q6L9LKSNU7EB:3EA3QWIZ4KYL82KAV49145N0S6YIT8", "A2TUUIV61CR0C7:33C7UALJVN1RACWOKZD0LAARXLN81Q", "A25FNSFSB048DL:3S0TNUHWKVLL27C00SXVMCB0MUKD83", "A3GWRDHAURRNK6:3SKEMFQBZ58TGDIAS9JIQP6Q6R9K8R", "A1ZE52NWZPN85P:3AAPLD8UCEKLC79QPMRG4TMLSJ7THU", "A34YDGVZKRJ0LZ:39U1BHVTDNU6IZ2RA12E0ZLB9E83TG", "A2TUUIV61CR0C7:39N5ACM9HGQU59Y0ATU4M2N0KYS9PJ", "A2DDPSXH2X96RF:3KKG4CDWKK18GGCHC92GJ4C5T8J49M", "AOG0PUCLMU0HH:3XUHV3NRVM1JR5Y0OQ9I1DG7CKFH5T", "A29CNQYJWMJ7G9:33LK57MYLV86OSW568SXUVU4M7BZS0", "A1NSHNH3MNFRGW:3LRLIPTPESC7Z1BPP73WMGCLWFEAKQ", "AIOOOO5OXWXKM:3RWE2M8QWJDC6UYAUIYJP2HCHV6N0U", "A29CNQYJWMJ7G9:3CP1TO84PV4FCFVI556BE9A5NGO25H", "A2TUUIV61CR0C7:3IUZPWIU1QA46EESQCZO459COABKWT", "AOMFEAWQHU3D8:34S9DKFK75S93PUV2Q9SHUBWT7RYNA", "A1ZE52NWZPN85P:3OS4RQUCRBI12PO3UACV1E4M6XYBFS", "AQ0YWEW29GSIJ:37M28K1J0SGCOH577M3KL1C8H5BJAR", "A2DDPSXH2X96RF:32N49TQG3ILLFC51OBH3OPN2C9RAVA", "AOMFEAWQHU3D8:3LWJHTCVCEPO6VQSDS9LW3ZLSXTFQ2", "AEWGY34WUIA32:32UTUBMZ7IZQYMATUPHZJ078SJ2BVE", "AKYXQY5IP7S0Z:3NLZY2D53RSA6N0OZ3CJRG45P32LQK", "A2DDPSXH2X96RF:3DPNQGW4LNILYXAJE2Z4ZUL33AU642", "A7XL1V3G7C2VV:3QY5DC2MXTNGYOX9U1TQ64WAVJ6UFV", "A2JQPSIVUCW92T:39GAF6DQWT3PLOS1SSOADOUZ8GRV1F", "A2TUUIV61CR0C7:3VE8AYVF8O0I0RQWRYSLACELYQ2F8K", "A2Q6L9LKSNU7EB:3OUYGIZWR91C9FANAXBBTRM7FNH0P1", "A2OFN0A5CPLH57:3QAPZX2QN6GGH89H8Z0ZXEEE6TP20E",]
    for file_path in files:
        data = load_dataset(file_path)
        scene_id = file_path.split("/")[-1].split(".")[0]
        episodes = []
        for episode in data["episodes"]:
            if episode["episode_id"] in episode_ids:
                episodes.append(episode)
        data["episodes"] = episodes
        if len(data["episodes"]) > 0:
            dest_path = os.path.join(output_path, "{}.json".format(scene_id))
            write_json(data, dest_path)
            write_gzip(dest_path, dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/hit_approvals/hits_max_length_1500.json"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/episodes/sample_hits.json"
    )
    parser.add_argument(
        "--per-scene-limit", type=int, default=10
    )
    parser.add_argument(
        "--limit", type=int, default=10
    )
    parser.add_argument(
        "--per-scene", dest='per_scene', action='store_true'
    )
    parser.add_argument(
        "--sample-episodes", dest='sample_episodes', action='store_true'
    )
    parser.add_argument(
        "--objectnav", dest='is_objectnav', action='store_true'
    )
    parser.add_argument(
        "--prev-tasks", type=str, default="data/datasets/objectnav_mp3d_v2/train/sampled/"
    )
    args = parser.parse_args()
    if args.sample_episodes and not args.is_objectnav and not args.per_scene:
        sample_episodes_by_episode_ids(args.input_path, args.output_path)
    elif args.sample_episodes and args.is_objectnav:
        sample_objectnav_episodes(args.input_path, args.output_path, args.prev_tasks)
    elif args.per_scene and args.sample_episodes:
        sample_episodes_by_scene(args.input_path, args.output_path, args.limit)
    else:
        sample_episodes(args.input_path, args.output_path, args.per_scene_limit)
