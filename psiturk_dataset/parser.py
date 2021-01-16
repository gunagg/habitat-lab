import argparse
import csv
import copy
import datetime
import glob
import gzip
import json
import re
import sys

from habitat.datasets.utils import VocabFromText


max_instruction_len = 9
instruction_list = []
unique_action_combo_map = {}
max_num_actions = 0
num_actions_lte_tenk = 0
total_episodes = 0
excluded_ep = 0

def read_csv(path, delimiter=","):
    file = open(path, "r")
    reader = csv.reader(file, delimiter=delimiter)
    return reader


def write_json(data, path):
    with open(path, 'w') as file:
        file.write(json.dumps(data))


def write_gzip(input_path, output_path):
    with open(input_path, "rb") as input_file:
        with gzip.open(output_path + ".gz", "wb") as output_file:
            output_file.writelines(input_file)


def column_to_json(col):
    if col is None:
        return None
    return json.loads(col)


def get_csv_rows(csv_reader):
    rows = []
    for row in csv_reader:
        rows.append(row)
    return rows


def is_viewer_step(data):
    if "type" in data.keys():
        if data["type"] == "runStep" and data["step"] == "viewer":
            return True
    return False


def preprocess(instruction):
    tokens = instruction.split()
    if len(tokens) < max_instruction_len:
        tokens = tokens + ["<pad>"] * (max_instruction_len - len(tokens))
    return " ".join(tokens)


def append_instruction(instruction):
    instruction_list.append(instruction)


def get_object_states(data):
    object_states = []
    for object_state in data["objectStates"]:
        object_states.append({
            "object_id": object_state["objectId"],
            "translation": object_state["translation"],
            "rotation": object_state["rotation"],
            "motion_type": object_state["motionType"],
        })
    return object_states


def get_action(data):
    if data is None:
        return None
    return data.get("action")


def is_physics_step(action):
    return (action == "stepPhysics")


def parse_replay_data_for_action(action, data):
    replay_data = {}
    replay_data["action"] = action
    if action == "grabReleaseObject":
        replay_data["is_grab_action"] = data["actionData"]["grabAction"]
        replay_data["is_release_action"] = data["actionData"]["releaseAction"]
        replay_data["object_under_cross_hair"] = data["actionData"]["objectUnderCrosshair"]
        replay_data["gripped_object_id"] = data["actionData"]["grippedObjectId"]

        action_data = {}

        if replay_data["is_release_action"]:
            action_data["new_object_translation"] = data["actionData"]["actionMeta"]["newObjectTranslation"]
            action_data["new_object_id"] = data["actionData"]["actionMeta"]["newObjectId"]
            action_data["object_handle"] = data["actionData"]["actionMeta"]["objectHandle"]
            action_data["gripped_object_id"] = data["actionData"]["actionMeta"]["grippedObjectId"]
        elif replay_data["is_grab_action"]:
            action_data["gripped_object_id"] = data["actionData"]["actionMeta"]["grippedObjectId"]

        replay_data["action_data"] = action_data
    else:
        replay_data["collision"] = data["collision"]
        replay_data["object_under_cross_hair"] = data["objectUnderCrosshair"]
        replay_data["nearest_object_id"] = data["nearestObjectId"]
        replay_data["gripped_object_id"] = data["grippedObjectId"]
    if "agentState" in data.keys():
        replay_data["agent_state"] = {
            "position": data["agentState"]["position"],
            "rotation": data["agentState"]["rotation"],
            "sensor_data": data["agentState"]["sensorData"]
        }
        replay_data["object_states"] = get_object_states(data)

    return replay_data


def parse_replay_data_for_step_physics(data):
    replay_data = {}
    replay_data["action"] = "stepPhysics"
    replay_data["object_under_cross_hair"] = data["objectUnderCrosshair"]
    #replay_data["object_drop_point"] = data["objectDropPoint"]
    if "agentState" in data.keys():
        replay_data["agent_state"] = {
            "position": data["agentState"]["position"],
            "rotation": data["agentState"]["rotation"],
            "sensor_data": data["agentState"]["sensorData"]
        }
    replay_data["object_states"] = get_object_states(data)
    return replay_data


def handle_step(step, episode, unique_id, timestamp):
    if step.get("event"):
        if step["event"] == "setEpisode":
            data = copy.deepcopy(step["data"]["episode"])
            episode["episode_id"] = unique_id
            episode["scene_id"] = data["sceneID"]
            episode["start_position"] = data["startState"]["position"]
            episode["start_rotation"] = data["startState"]["rotation"]

            episode["objects"] = []
            for idx in range(len(data["objects"])):
                object_data = {}
                object_data["object_id"] = data["objects"][idx]["objectId"]
                object_data["object_template"] = data["objects"][idx]["objectHandle"]
                object_data["position"] = data["objects"][idx]["position"]
                object_data["rotation"] = data["objects"][idx]["rotation"]
                object_data["motion_type"] = data["objects"][idx]["motionType"]
                object_data["object_icon"] = data["objects"][idx]["objectIcon"]
                object_data["is_receptacle"] = data["objects"][idx]["isReceptacle"]
                episode["objects"].append(object_data)

            instruction_text = data["task"]["instruction"]
            episode["instruction"] = {
                "instruction_text": instruction_text,
                #"preprocessed_instruction": preprocess(instruction_text)
            }
            append_instruction(instruction_text)
            object_receptacle_map = {}
            if "goals" in data["task"].keys():
                object_receptacle_map = data["task"]["goals"]["objectToReceptacleMap"]
            episode["goals"] = {
                "object_receptacle_map": object_receptacle_map
            }
            episode["reference_replay"] = []

        elif step["event"] == "handleAction":
            data = parse_replay_data_for_action(step["data"]["action"], step["data"])
            data["timestamp"] = timestamp
            episode["reference_replay"].append(data)

        elif is_physics_step(step["event"]):
            data = parse_replay_data_for_step_physics(step["data"])
            data["timestamp"] = timestamp
            episode["reference_replay"].append(data)

    elif step.get("type"):
        if step["type"] == "finishStep":
            return True
    return False


def convert_to_episode(csv_reader):
    episode = {}
    viewer_step = False
    start_ts = 0
    end_ts = 0
    for row in csv_reader:
        unique_id = row[0]
        step = row[1]
        timestamp = row[2]
        data = column_to_json(row[3])

        if start_ts == 0:
            start_ts = int(timestamp)

        if not viewer_step:
            viewer_step = is_viewer_step(data)

        if viewer_step:
            is_viewer_step_finished = handle_step(data, episode, unique_id, timestamp)
        end_ts = int(timestamp)
    
    episode["reference_replay"] = post_process_episode(copy.deepcopy(episode["reference_replay"]))
    start_dt = datetime.datetime.fromtimestamp(start_ts / 1000)
    end_dt = datetime.datetime.fromtimestamp(end_ts / 1000)

    global max_num_actions
    global total_episodes
    global num_actions_lte_tenk
    global excluded_ep

    ep_len_in_sec = (end_dt - start_dt).total_seconds()
    max_num_actions += ep_len_in_sec
    if len(episode["reference_replay"]) < 9500:
        num_actions_lte_tenk += ep_len_in_sec
        excluded_ep += 1
    total_episodes += 1
    return episode


def merge_replay_data_for_action(action_data_list):
    if len(action_data_list) == 1:
        return action_data_list[0]

    first_action_data = action_data_list[0]
    action = first_action_data["action"]
    last_action_data = action_data_list[-1]

    if len(action_data_list) == 2:
        last_action_data["action"] = action
        if action == "grabReleaseObject":
            last_action_data["action_data"] = first_action_data["action_data"]
            last_action_data["is_grab_action"] = first_action_data["is_grab_action"]
            last_action_data["is_release_action"] = first_action_data["is_release_action"]
            last_action_data["object_under_cross_hair"] = first_action_data["object_under_cross_hair"]
            last_action_data["gripped_object_id"] = first_action_data["gripped_object_id"]
        else:
            last_action_data["collision"] = first_action_data["collision"]
            last_action_data["object_under_cross_hair"] = first_action_data["object_under_cross_hair"]
            last_action_data["nearest_object_id"] = first_action_data["nearest_object_id"]
        return last_action_data

    if len(action_data_list) >= 3:
        print("\n\n\nIncorrectly aligned actions in episode")
        sys.exit(1)
    return None


def post_process_episode(reference_replay):
    i = 0
    post_processed_ref_replay = []
    unique_action_combo_map = {}
    while i < len(reference_replay):
        data = reference_replay[i]
        action = get_action(data)

        if not is_physics_step(action):
            old_i = i
            action_data_list = [data]
            while i < len(reference_replay) and not is_physics_step(get_action(data)):
                data = reference_replay[i + 1]
                action_data_list.append(data)
                i += 1
            data = merge_replay_data_for_action(copy.deepcopy(action_data_list))
            if len(action_data_list) == 3:
                action_str = "".join([dd.get("action") for dd in action_data_list])
                if not data["action"] in unique_action_combo_map.keys():
                    unique_action_combo_map[data["action"]] = 0
                unique_action_combo_map[data["action"]] += 1

        post_processed_ref_replay.append(data)
        i += 1
    print(unique_action_combo_map, len(post_processed_ref_replay), len(reference_replay))
    return post_processed_ref_replay


def compute_instruction_tokens(episodes):
    instruction_vocab = VocabFromText(
        sentences=list(set(instruction_list))
    )
    max_token_size = 0
    for episode in episodes:
        instruction = episode["instruction"]["instruction_text"]
        instruction_tokens = instruction_vocab.tokenize_and_index(instruction, keep=())
        max_token_size = max(max_token_size, len(instruction_tokens))

    for episode in episodes:
        instruction = episode["instruction"]["instruction_text"]
        instruction_tokens = instruction_vocab.tokenize_and_index(instruction, keep=())
        if len(instruction_tokens) < max_token_size:
            instruction_tokens = instruction_tokens + [instruction_vocab.word2idx("<pad>")] * (max_token_size - len(instruction_tokens))
        episode["instruction"]["instruction_tokens"] = instruction_tokens
    return episodes


def replay_to_episode(replay_path, output_path, max_episodes=1):
    global max_num_actions
    global total_episodes
    global num_actions_lte_tenk
    global excluded_ep

    all_episodes = {
        "episodes": []
    }

    episodes = []
    file_paths = glob.glob(replay_path + "/*.csv")
    file_paths = file_paths[:max_episodes]
    for file_path in file_paths:
        print(file_path)
        reader = read_csv(file_path)
        episode = convert_to_episode(reader)
        episodes.append(episode)

    all_episodes["episodes"] = compute_instruction_tokens(copy.deepcopy(episodes))
    all_episodes["instruction_vocab"] = {
        "sentences": list(set(instruction_list))
    }
    print("unique action combo map:\n")
    for key, v in unique_action_combo_map.items():
        print(key)

    print("Total episodes: {}".format(len(all_episodes["episodes"])))
    print("Average time per episode: {}, Total actions: {} -- {}".format(max_num_actions / total_episodes, max_num_actions, total_episodes))
    print("(Exclude > 15 min) Average time per episode: {}, Total actions: {} -- {}".format(num_actions_lte_tenk/excluded_ep, num_actions_lte_tenk, excluded_ep))
    write_json(all_episodes, output_path)
    write_gzip(output_path, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-path", type=str, default="data/hit_data"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/episodes/data.json"
    )
    parser.add_argument(
        "--max-episodes", type=int, default=1
    )
    args = parser.parse_args()
    replay_to_episode(args.replay_path, args.output_path, args.max_episodes)


if __name__ == '__main__':
    main()




