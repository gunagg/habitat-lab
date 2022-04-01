import argparse
import json

def load_episode(path="task_1_episode.json"):
	with open(path, "r") as json_file:
		data = json.loads(json_file.read())
		return data


def extract_object_points(data):
	object_points = []
	for object_ in data["objects"]:
		object_points.append(object_["position"])
	return object_points


def extract_agent_point(data):
	return data["start_position"]


def validate(path):
	data = load_episode(path)
	points_dict = {}

	for episode in data["episodes"]:
		points = []
		points.append(extract_agent_point(episode))

		points.extend(extract_object_points(episode))
		for point in points:
			if points_dict.get(str(point)) is not None:
				print("\n Invalid Episode!!!\n")
				return
			else:
				points_dict[str(point)] = 1
		print("\n Total points: {}".format(len(points)))
		print("\n Unique points: {}".format(len(points_dict.values())))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
        description="Generate a new messy scene."
    )
	parser.add_argument(
        "--episode",
        default="task_1_episode.json",
        help="Epsiode configuration",
    )

	args = parser.parse_args()

	validate(args.episode)

