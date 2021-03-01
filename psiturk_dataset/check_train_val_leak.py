import argparse
import gzip
import json


VISITED_POINT_DICT = {}
assignment_dict = {}
episode_ids = ['A1L3937MY09J3I:3Z7EFSHGNBH1CG7U84ECI5ABGYYCX5','A1ZE52NWZPN85P:3C6FJU71TSWMYFE4ZRLEVP3QQU9YUY','A2CWA5VQZ6IWMQ:3YGXWBAF72KAEEJKOTC7LUDDNIP4C8','APGX2WZ59OWDN:358010RM5GWXBPDUZL9H8XY015IVXR']

def load_dataset(path):
    with gzip.open(path, "rb") as file:
        data = json.loads(file.read(), encoding="utf-8")
    return data


def load_json_dataset(path):
    file = open(path, "r")
    data = json.loads(file.read())
    return data


def validate_data(input_path_1, input_path_2):
    train_data = load_dataset(input_path_1)
    eval_data = load_dataset(input_path_2)

    train_episode_ids = []
    train_instructions = []
    train_scene_map = {}
    for episode in train_data["episodes"]:
        train_episode_ids.append(episode["episode_id"])
        train_instructions.append(episode["instruction"]["instruction_text"])
        scene_id = episode["scene_id"]
        if scene_id not in train_scene_map.keys():
            train_scene_map[scene_id] = 0
        train_scene_map[scene_id] += 1

    eval_episode_ids = []
    eval_instructions = []
    eval_scene_map = {}
    for episode in eval_data["episodes"]:
        eval_episode_ids.append(episode["episode_id"])
        eval_instructions.append(episode["instruction"]["instruction_text"])
        scene_id = episode["scene_id"]
        if scene_id not in eval_scene_map.keys():
            eval_scene_map[scene_id] = 0
        eval_scene_map[scene_id] += 1

    print("\nOverlap episodes: {}".format(len(set(train_episode_ids).intersection(set(eval_episode_ids)))))
    print("Unique train episodes: {}".format(len(set(train_episode_ids))))
    print("Unique eval episodes: {}".format(len(set(eval_episode_ids))))

    print("\nOverlap instructions: {}".format(len(set(train_instructions).intersection(set(eval_instructions)))))
    print("Unique instructions train episodes: {}".format(len(set(train_instructions))))
    print("Unique instructions eval episodes: {}".format(len(set(eval_instructions))))

    print("\nTrain data scene map: {}".format(train_scene_map))
    print("Eval data scene map: {}".format(eval_scene_map))


def populate_points(episodes, populate=True):
    redundant_episode_ids = []
    single_point_duplicate_episodes = []
    for episode in episodes:
        point = str(episode["start_position"])
        instruction = episode["instruction"]["instruction_text"]
        scene_id =  episode["scene_id"]
        episode_id =  episode["episode_id"]
        unique_key = "{}:{}".format(scene_id, instruction)
        duplicate_agent_point = False
        if VISITED_POINT_DICT.get(point):
            if populate:
                VISITED_POINT_DICT[point]["instruction"].append(instruction)
                VISITED_POINT_DICT[point]["scene"].append(scene_id)
            # print("Redundant agent position in episode {}".format(episode["episode_id"]))
            duplicate_agent_point = True
        else:
            if populate:
                VISITED_POINT_DICT[point] = {
                    "instruction": [instruction],
                    "scene": [scene_id]
                }

        duplicate_object_points = True
        for object_ in episode["objects"]:
            point = str(object_["position"])
            if VISITED_POINT_DICT.get(point):
                if populate:
                    VISITED_POINT_DICT[point]["instruction"].append(instruction)
                    VISITED_POINT_DICT[point]["scene"].append(scene_id)
                # print("Redundant point in episode {}".format(episode["episode_id"]))
            else:
                if populate:
                    VISITED_POINT_DICT[point] = {
                        "instruction": [instruction],
                        "scene": [scene_id]
                    }
                duplicate_object_points = False
        
        is_trajectory_duplicate = False
        trajectory_length = len(episode["reference_replay"])
        if not unique_key in assignment_dict.keys():
            assignment_dict[unique_key]= {
                "reference_replay_length": [],
                "episode_ids": []
            }
        else:
            stored_trajectory_lens = assignment_dict[unique_key]
            if trajectory_length in stored_trajectory_lens:
                is_trajectory_duplicate = True
        assignment_dict[unique_key]["reference_replay_length"].append(trajectory_length)
        assignment_dict[unique_key]["episode_ids"].append(episode_id)
        
        if duplicate_agent_point and duplicate_object_points:
            redundant_episode_ids.append({
                "instruction": instruction,
                "scene_id": scene_id,
                "episode_id": episode_id
            })
        if is_trajectory_duplicate and duplicate_agent_point and duplicate_object_points:
            single_point_duplicate_episodes.append({
                "instruction": instruction,
                "scene_id": scene_id,
                "episode_id": episode_id,
            })
    return redundant_episode_ids, single_point_duplicate_episodes


def validate_episode_init_leak(input_path_1, input_path_2):
    train_data = load_dataset(input_path_1)
    eval_data = load_dataset(input_path_2)

    train_episodes = train_data["episodes"]
    eval_episodes = eval_data["episodes"]

    redundant_points_train_episodes, single_point_train_duplicate_episodes = populate_points(train_episodes)
    print("\n\nEpisode leak in train episodes: {}".format(len(redundant_points_train_episodes)))
    print("\n\Init points leak in train episodes: {}".format(len(single_point_train_duplicate_episodes)))
    print("\n\nEpisodes: {}".format(redundant_points_train_episodes))
    print("\n\nEpisodes: {}".format(single_point_train_duplicate_episodes))
    redundant_points_eval_episodes, single_point_eval_duplicate_episodes = populate_points(eval_episodes, populate=False)
    print("\n\Episode leak in eval episodes: {}".format(len(redundant_points_eval_episodes)))
    print("\n\Init points leak in eval episodes: {}".format(len(single_point_eval_duplicate_episodes)))
    print("\n\nEpisodes: {}".format(redundant_points_eval_episodes))
    print("\n\nEpisodes: {}".format(single_point_eval_duplicate_episodes))
    print("\n\n")
    for dup in single_point_eval_duplicate_episodes:
        print(dup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path-1", type=str, default="data/hit_approvals/dataset/backup/train.json.gz"
    )
    parser.add_argument(
        "--input-path-2", type=str, default="data/datasets/object_rearrangement/v0/train/train.json.gz"
    )
    parser.add_argument(
        "--check-leak", dest='check_leak', action='store_true'
    )
    args = parser.parse_args()


    if args.check_leak:
        validate_episode_init_leak(args.input_path_1, args.input_path_2)
    else:
        validate_data(args.input_path_1, args.input_path_2)


if __name__ == "__main__":
    main()
