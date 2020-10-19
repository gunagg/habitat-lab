import argparse
import csv
import copy
import json


def read_csv(path, delimiter=","):
    file = open(path, "r")
    reader = csv.reader(file, delimiter=delimiter)
    return reader


def write_json(data, path):
    with open(path, 'w') as file:
        file.write(json.dumps(data))


def column_to_json(col):
    return json.loads(col)


def is_viewer_step(data):
    if "type" in data.keys():
        if data["type"] == "runStep" and data["step"] == "viewer":
            return True
    return False


def handle_step(step, episode, episode_id, timestamp, prev_timestamp):

    if step.get("event"):
        if step["event"] == "setEpisode":
            data = copy.deepcopy(step["data"]["episode"])
            episode["episode_id"] = episode_id
            episode["scene_id"] = data["sceneID"]
            episode["start_position"] = data["startState"]["position"]
            episode["start_rotation"] = data["startState"]["rotation"]

            episode["objects"] = []
            for idx in range(len(data["objects"])):
                object_data = {}
                object_data["object_id"] = idx
                object_data["object_template"] = data["objects"][idx]["objectHandle"]
                object_data["position"] = data["objects"][idx]["position"]
                object_data["motion_type"] = data["objects"][idx]["motionType"]
                object_data["object_icon"] = data["objects"][idx]["objectIcon"]
                episode["objects"].append(object_data)

            episode["instruction"] = {
                "instruction_text": data["task"]["instruction"]
            }
            object_receptacle_map = {}
            if "goals" in data["task"].keys():
                object_receptacle_map = data["task"]["goals"]["objectToReceptacleMap"]
            episode["goals"] = {
                "object_receptacle_map": object_receptacle_map
            }
            episode["reference_replay"] = []
        elif step["event"] == "handleAction":
            step["data"]["timestamp"] = timestamp
            step["data"]["prev_timestamp"] = prev_timestamp
            episode["reference_replay"].append(step["data"])
        elif (step["event"] == "stepPhysics"):
            data = copy.deepcopy(step["data"])
            data["action"] = "stepPhysics"
            data["object_states"] = []
            for object_state in step["data"]["objectStates"]:
                data["object_states"].append({
                    "object_id": object_state["objectId"],
                    "translation": object_state["translation"],
                    "rotation": object_state["rotation"],
                    "motion_type": object_state["motionType"],
                })
            data.pop("objectStates")
            data.pop("step")
            episode["reference_replay"].append(data)
    elif step.get("type"):
        if step["type"] == "finishStep":
            return True
    return False



def convert_to_episode(csv_reader):
    episode = {}
    viewer_step = False
    prev_timestamp = None
    for row in csv_reader:
        episode_id = row[0]
        step = row[1]
        timestamp = row[2]
        data = column_to_json(row[3])

        if prev_timestamp is None:
            prev_timestamp = timestamp

        if not viewer_step:
            viewer_step = is_viewer_step(data)

        if viewer_step:
            is_viewer_step_finished = handle_step(data, episode, 0, timestamp, prev_timestamp)
        prev_timestamp = timestamp

    return episode


def replay_to_episode(replay_path, output_path):
    reader = read_csv(replay_path)
    episode = convert_to_episode(reader)
    all_episodes = {
        "episodes": [episode]
    }
    write_json(all_episodes, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-file", type=str, default="replays/demo_1.csv"
    )
    parser.add_argument(
        "--output-file", type=str, default="replays/demo_1.json"
    )
    args = parser.parse_args()
    print(args)
    replay_to_episode(args.replay_file, args.output_file)


if __name__ == '__main__':
    main()




