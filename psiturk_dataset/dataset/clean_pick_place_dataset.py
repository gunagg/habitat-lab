import argparse

from psiturk_dataset.utils.utils import load_dataset, write_json, write_gzip


def clean_dataset(input_path, output_path):
    dataset = load_dataset(input_path)
    for episode in dataset["episodes"]:
        for step in episode["reference_replay"]:
            action = step["action"]
            is_grab_action = step.get("is_grab_action")
            is_release_action = step.get("is_release_action")

            if action == "GRAB_RELEASE" and step.get("action_data") is not None and len(step["action_data"].keys()) > 0:
                if is_release_action:
                    step["action_data"]["released_object_id"] = step["action_data"]["gripped_object_id"]
                    step["action_data"]["released_object_handle"] = step["action_data"]["object_handle"]
                    step["action_data"]["released_object_position"] = step["action_data"]["new_object_translation"]
                    del step["action_data"]["object_handle"]
                    del step["action_data"]["new_object_translation"]
                    del step["action_data"]["new_object_id"]
                elif is_grab_action:
                    step["action_data"]["grab_object_id"] = step["action_data"]["gripped_object_id"]
                del step["action_data"]["gripped_object_id"]
            if "timestamp" in step.keys():
                del step["timestamp"]
            if "nearest_object_id" in step.keys():
                del step["nearest_object_id"]

    write_json(dataset, output_path)
    write_gzip(output_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/episodes/data.json"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/episodes/data.json"
    )
    args = parser.parse_args()
    clean_dataset(args.input_path, args.output_path)
