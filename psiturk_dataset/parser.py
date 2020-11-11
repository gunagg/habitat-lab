import argparse
import csv
import copy
import glob
import gzip
import json


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
    return json.loads(col)


def is_viewer_step(data):
    if "type" in data.keys():
        if data["type"] == "runStep" and data["step"] == "viewer":
            return True
    return False


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

    return replay_data


def parse_replay_data_for_step_physics(data):
    replay_data = {}
    replay_data["action"] = "stepPhysics"
    replay_data["object_under_cross_hair"] = data["objectUnderCrosshair"]
    replay_data["object_states"] = []
    for object_state in data["objectStates"]:
        replay_data["object_states"].append({
            "object_id": object_state["objectId"],
            "translation": object_state["translation"],
            "rotation": object_state["rotation"],
            "motion_type": object_state["motionType"],
        })
    return replay_data


def handle_step(step, episode, episode_id):

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
                object_data["object_id"] = data["objects"][idx]["objectId"]
                object_data["object_template"] = data["objects"][idx]["objectHandle"]
                object_data["position"] = data["objects"][idx]["position"]
                object_data["rotation"] = data["objects"][idx]["rotation"]
                object_data["motion_type"] = data["objects"][idx]["motionType"]
                object_data["object_icon"] = data["objects"][idx]["objectIcon"]
                object_data["is_receptacle"] = data["objects"][idx]["isReceptacle"]
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
            data = parse_replay_data_for_action(step["data"]["action"], step["data"])
            episode["reference_replay"].append(data)

        elif (step["event"] == "stepPhysics"):
            data = parse_replay_data_for_step_physics(step["data"])
            episode["reference_replay"].append(data)

    elif step.get("type"):
        if step["type"] == "finishStep":
            return True
    return False



def convert_to_episode(csv_reader):
    episode = {}
    viewer_step = False
    for row in csv_reader:
        episode_id = row[0]
        step = row[1]
        timestamp = row[2]
        data = column_to_json(row[3])

        if not viewer_step:
            viewer_step = is_viewer_step(data)

        if viewer_step:
            is_viewer_step_finished = handle_step(data, episode, 0)

    return episode


def replay_to_episode(replay_path, output_path):
    all_episodes = {
        "episodes": []
    }
    for file_path in glob.glob(replay_path + "/*.csv"):
        print(file_path)
        reader = read_csv(file_path)
        episode = convert_to_episode(reader)
        all_episodes["episodes"].append(episode)

    print("Total episodes: {}".format(len(all_episodes["episodes"])))
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
    args = parser.parse_args()
    replay_to_episode(args.replay_path, args.output_path)


if __name__ == '__main__':
    main()




