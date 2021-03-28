import argparse
import csv
import copy
import datetime
import glob
import gzip
import json
import re
import sys

from tqdm import tqdm
from habitat.datasets.utils import VocabFromText


max_instruction_len = 9
instruction_list = []
unique_action_combo_map = {}
max_num_actions = 0
num_actions_lte_tenk = 0
total_episodes = 0
excluded_ep = 0
task_episode_map = {}
filter_episodes = ["A1NSHNH3MNFRGW:39L1G8WVWSU59FQI8II4UT2G6QI13L", "A2CWA5VQZ6IWMQ:39U1BHVTDNU6IZ2RA12E0ZLBY9LT3Y", "A1NSHNH3MNFRGW:3EFVCAY5L5CY5TCSAOJ6PA6DGTD8JR", "A1ZE52NWZPN85P:3QY5DC2MXTNGYOX9U1TQ64WAKDYFUL", "AV0PUPRI47UDT:3CN4LGXD5ZRNHHKPKLUWIL5WRBM4Y6", "A1NSHNH3MNFRGW:3EO896NRAYYH3D4GDMU1G620U6UTJX", "A1ZE52NWZPN85P:3OONKJ5DKEMV821WTDVLO8D0NTABOE", "A2CWA5VQZ6IWMQ:3CN4LGXD5ZRNHHKPKLUWIL5WRBNY41", "A2CWA5VQZ6IWMQ:3VD82FOHKSREI7T27DRGZSJI5UOCOW", "A1ZE52NWZPN85P:3EO896NRAYYH3D4GDMU1G620UL7TJ4", "A2CWA5VQZ6IWMQ:35H6S234SC33UGEJS7IE4MRHS3M65N", "AKYXQY5IP7S0Z:3JW0YLFXRVJV1E89FQIRSG3706JWW3", "A1NSHNH3MNFRGW:3GNCZX450KQ8AS852Z84IXYKFQBPAO", "A3O5RKGH6VB19C:38F71OA9GVZXLGS0LZ24FUFGBIRMFI", "A3KC26Z78FBOJT:3QBD8R3Z23MBN3GNEYLYGU7UIH34OC", "A3O5RKGH6VB19C:39K0FND3AJI2PPBSAJGC1T4PE79AMY", "A2Q6L9LKSNU7EB:3VSOLARPKDCNYKTDCVXX9ZKZ8TB93T", "A272X64FOZFYLB:33M4IA01QI45IIWDQ147709XL9WRXA", "A2Q6L9LKSNU7EB:3LOZAJ85YFGOEYFSBBP66S1PA1O2XJ", "AKYXQY5IP7S0Z:3CFVK00FWNOHW5H4KUYLLBNEJXKL61", "A2Q6L9LKSNU7EB:3KMS4QQVK4T2VSSX0NPO0HNCMECFKO", "A3PFU4042GIQLE:34Z02EIMIUGA173UREKVY1N4026T0N", "AEWGY34WUIA32:3WYGZ5XF3YIBZXXJ67PN7G6RB28KSA", "A2Q6L9LKSNU7EB:3180JW2OT6FFIBTQCQC3DQWMI02J5O", "A1ZE52NWZPN85P:3ZPPDN2SLXZQ8I9A1FETSQOWVR5E97", "ADXHWQLUQBK77:3TK8OJTYM3OS2GB3DUZ0EKCXZLLVP9", "A272X64FOZFYLB:3J2UYBXQQNF4Z9SIV1C2NRVQBPE60T", "A1ZE52NWZPN85P:3IX2EGZR7DM4NYRO9XP6GR1I61HJRQ", "AEWGY34WUIA32:39O5D9O87VVPWI0GOF7OBPL79IXC3Z", "A1ZE52NWZPN85P:3X0H8UUIT3R2UXR0VL8QVR0MUY2SW9", "A2CWA5VQZ6IWMQ:31QTRG6Q2VG96A68I5MKLJGRI7NYPW"]


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


def remap_action(action):
    if action == "turnRight":
        return "TURN_RIGHT"
    elif action == "turnLeft":
        return "TURN_LEFT"
    elif action == "moveForward":
        return "MOVE_FORWARD"
    elif action == "moveBackward":
        return "MOVE_BACKWARD"
    elif action == "lookUp":
        return "LOOK_UP"
    elif action == "lookDown":
        return "LOOK_DOWN"
    elif action == "grabReleaseObject":
        return "GRAB_RELEASE"
    elif action == "stepPhysics":
        return "NO_OP"
    return "STOP"


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
            task_episode_map[data["sceneID"]].append(int(data["episodeID"]))

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
            if "place the shark" in instruction_text:
                instruction_text = instruction_text.replace("place the shark", "place the toy shark")
            episode["instruction"] = {
                "instruction_text": instruction_text.lower(),
            }
            append_instruction(instruction_text)
            object_receptacle_map = {}
            if "goals" in data["task"].keys():
                object_receptacle_map = data["task"]["goals"]["objectToReceptacleMap"]
                goals = []
                for object_ in episode["objects"]:
                    goals.append({
                        "position": object_["position"],
                        "rotation": object_["rotation"],
                        "info": {
                            "is_receptacle": object_["is_receptacle"],
                        }
                    })
                episode["goals"] = goals
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
    
    actual_episode_length = len(episode["reference_replay"])
    post_processed_ref_replay = post_process_episode(copy.deepcopy(episode["reference_replay"]))
    pruned_reference_replay = prune_episode_end(copy.deepcopy(post_processed_ref_replay))
    episode["reference_replay"] = append_episode_start_and_end_steps(copy.deepcopy(pruned_reference_replay), episode)

    post_processed_episode_length = len(post_processed_ref_replay)
    pruned_episode_length = len(episode["reference_replay"])    

    start_dt = datetime.datetime.fromtimestamp(start_ts / 1000)
    end_dt = datetime.datetime.fromtimestamp(end_ts / 1000)
    hit_duration = (end_dt - start_dt).total_seconds()

    episode_length = {
        "actual_episode_length": actual_episode_length,
        "post_processed_episode_length": post_processed_episode_length,
        "pruned_episode_length": pruned_episode_length,
        "hit_duration": hit_duration
    }
    return episode, episode_length


def append_episode_start_and_end_steps(reference_replay, episode):
    stop_step = copy.deepcopy(reference_replay[-1])
    stop_step["action"] = "STOP"

    start_step = copy.deepcopy(reference_replay[0])
    start_step["action"] = "STOP"

    for object_ in episode["objects"]:
        for step_object_state in start_step["object_states"]:
            if step_object_state["object_id"] == object_["object_id"]:
                step_object_state["translation"] = object_["position"]
                step_object_state["rotation"] = object_["rotation"]
                break
    
    reference_replay = [start_step] + reference_replay + [stop_step]
    # print("Add start/stop action replay size: {}".format(len(reference_replay)))
    return reference_replay[:]


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
    return post_processed_ref_replay


def is_redundant_state_action_pair(current_state, prev_state):
    if prev_state is None:
        return False
    current_state = copy.deepcopy(current_state)
    prev_state = copy.deepcopy(prev_state)
    del current_state["timestamp"]
    del prev_state["timestamp"]
    current_state_json_string = json.dumps(current_state)
    prev_state_json_string = json.dumps(prev_state)
    return current_state_json_string == prev_state_json_string


def prune_episode_end(reference_replay):
    pruned_reference_replay = []
    prev_state = None
    redundant_state_count = 0
    for i in range(len(reference_replay)):
        data = reference_replay[i]
        if "action" in data.keys() and is_physics_step(get_action(data)) and not is_redundant_state_action_pair(data, prev_state):
            copy_data = copy.deepcopy(data)
            copy_data["action"] = remap_action(data["action"])
            pruned_reference_replay.append(copy_data)
        elif "action" in data.keys() and not is_physics_step(get_action(data)):
            copy_data = copy.deepcopy(data)
            copy_data["action"] = remap_action(data["action"])
            pruned_reference_replay.append(copy_data)
        else:
            redundant_state_count += 1

        prev_state = copy.deepcopy(data)

    # print("Original replay size: {}, pruned replay: {}, redundant steps: {}".format(len(reference_replay), len(pruned_reference_replay), redundant_state_count))
    # Add action buffer for 3 seconds
    # 3 seconds is same as interface but we can try reducing it
    reference_replay = pruned_reference_replay[:]
    return reference_replay


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


def replay_to_episode(replay_path, output_path, max_episodes=16,  max_episode_length=1500, sample=False):
    all_episodes = {
        "episodes": []
    }

    episodes = []
    episode_lengths = []
    file_paths = glob.glob(replay_path + "/*.csv")
    for file_path in tqdm(file_paths):
        reader = read_csv(file_path)
        episode, counts = convert_to_episode(reader)
        # Filter out episodes that have unstable initialization
        if episode["episode_id"] in filter_episodes:
            continue
        if len(episode["reference_replay"]) <= max_episode_length:
            episodes.append(episode)
            episode_lengths.append(counts)
        if sample:
            if len(episodes) >= max_episodes:
                break


    all_episodes["episodes"] = compute_instruction_tokens(copy.deepcopy(episodes))
    all_episodes["instruction_vocab"] = {
        "sentences": list(set(instruction_list))
    }

    if len(unique_action_combo_map.keys()) > 0:
        print("unique action combo map:\n")
        for key, v in unique_action_combo_map.items():
            print(key)

    show_average(all_episodes, episode_lengths)
    write_json(all_episodes, output_path)
    write_gzip(output_path, output_path)


def show_average(all_episodes, episode_lengths):
    print("Total episodes: {}".format(len(all_episodes["episodes"])))

    total_episodes = len(all_episodes["episodes"])
    total_hit_duration = 0
    total_hit_duration_filtered = 0
    filtered_episode_count = 0

    total_actions = 0
    total_actions_postprocessed = 0
    total_actions_pruned = 0
    total_actions_filtered = 0
    total_actions_postprocessed_filtered = 0
    total_actions_pruned_filtered = 0
    num_eps_gt_than_2k = 0
    for episode_length  in episode_lengths:
        total_hit_duration += episode_length["hit_duration"]
        total_actions += episode_length["actual_episode_length"]
        total_actions_postprocessed += episode_length["post_processed_episode_length"]
        total_actions_pruned += episode_length["pruned_episode_length"]
        
        if episode_length["pruned_episode_length"] < 5000:
            total_hit_duration_filtered += episode_length["hit_duration"]
            total_actions_filtered += episode_length["actual_episode_length"]
            total_actions_postprocessed_filtered += episode_length["post_processed_episode_length"]
            total_actions_pruned_filtered += episode_length["pruned_episode_length"]
            filtered_episode_count += 1
        num_eps_gt_than_2k += 1 if episode_length["pruned_episode_length"] > 1900 else 0

    print("\n\n")
    print("Average hit duration")
    print("All hits: {}, Total duration: {}, Num episodes: {}".format(round(total_hit_duration / total_episodes, 2), total_hit_duration, total_episodes))
    print("Filtered hits: {}, Total duration: {}, Num episodes: {}".format(round(total_hit_duration_filtered / filtered_episode_count, 2), total_hit_duration_filtered, filtered_episode_count))
    
    print("\n\n")
    print("Average episode length:")
    print("All Hits: {}, Num actions: {}, Num episodes: {}".format(round(total_actions / total_episodes, 2), total_actions, total_episodes))
    print("Filtered Hits: {}, Num actions: {}, Num episodes {}".format(round(total_actions_filtered / filtered_episode_count, 2), total_actions_filtered, filtered_episode_count))

    print("\n\n")
    print("Average postprocessed episode length:")
    print("All Hits: {}, Num actions: {}, Num episodes: {}".format(round(total_actions_postprocessed / total_episodes, 2), total_actions_postprocessed, total_episodes))
    print("Filtered Hits: {}, Num actions: {}, Num episodes {}".format(round(total_actions_postprocessed_filtered / filtered_episode_count, 2), total_actions_postprocessed_filtered, filtered_episode_count))

    print("\n\n")
    print("Average pruned episode length:")
    print("All Hits: {}, Num actions: {}, Num episodes: {}".format(round(total_actions_pruned / total_episodes, 2), total_actions_pruned, total_episodes))
    print("Filtered Hits: {}, Num actions: {}, Num episodes {}".format(round(total_actions_pruned_filtered / filtered_episode_count, 2), total_actions_pruned_filtered, filtered_episode_count))

    print("\n\n")
    print("Pruned episodes greater than 1.9k actions: {}".format(num_eps_gt_than_2k))


def list_missing_episodes():
    episode_ids = set([i for i in range(1371)])
    for key, val in task_episode_map.items():
        val_set = set([int(v) for v in val])
        missing_episodes = episode_ids.difference(val_set)
        print("Missing episodes for scene: {} are: {}".format(key, len(list(missing_episodes))))
    write_json(task_episode_map, "data/hit_data/complete_task_map.json")


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
    parser.add_argument(
        "--max-episode-length", type=int, default=15000
    )
    args = parser.parse_args()
    replay_to_episode(args.replay_path, args.output_path, args.max_episodes, args.max_episode_length)
    list_missing_episodes()


if __name__ == '__main__':
    main()


