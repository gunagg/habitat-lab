import gzip
import json


def write_json(data, path):
    with open(path, 'w') as file:
        file.write(json.dumps(data))


def write_gzip(input_path, output_path):
    with open(input_path, "rb") as input_file:
        with gzip.open(output_path + ".gz", "wb") as output_file:
            output_file.writelines(input_file)


def save_frame(
    frame, path: str
) -> None:
    r"""For saving RGB reconstruction results during EQA-CNN-Pretrain eval.

    Args:
        gt_rgb: RGB ground truth tensor
        pred_rgb: RGB reconstruction tensor
        path: to save images
    """
    im = Image.fromarray(frame)
    im.save(path + ".png")

