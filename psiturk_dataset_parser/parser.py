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

	if (step.get("event")):
		if (step["event"] == "setEpisode"):
			data = step["data"]["episode"]
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
			episode["goals"] = {
				"object_receptacle_map": data["task"]["goals"]["objectToReceptacleMap"]
			}
			episode["reference_replay"] = []
		elif (step["event"] == "handleAction"):
			step["data"]["timestamp"] = timestamp
			step["data"]["prev_timestamp"] = prev_timestamp
			episode["reference_replay"].append(step["data"])
	return



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
			handle_step(data, episode, 0, timestamp, prev_timestamp)
		prev_timestamp = timestamp
			
	return episode


if __name__ == '__main__':
	reader = read_csv('demo.csv')
	episode = convert_to_episode(reader)
	all_episodes = {
		"episodes": [episode]
	}
	write_json(all_episodes, 'demo.json')




