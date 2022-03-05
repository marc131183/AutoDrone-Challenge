import os
import numpy as np
import optuna

from PIL import Image
from typing import List, Dict, Tuple


def openLabelFile(path: str) -> List[Tuple[float, float, float, float, float]]:
    """
    returns a list with all bounding boxes in the image: label, x, y, width, height of the bounding box
    x, y, width, height are returned in [0, 1]
    """
    with open(path, "r") as f:
        lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = [float(elem) for elem in lines[i][:-1].split(" ")]

    return lines


def computeScoreForImage(
    img_path: str,
    label: List[Tuple[float, float, float, float, float]],
    r_min: int,
    r_max: int,
    g_min: int,
    g_max: int,
    b_min: int,
    b_max: int,
) -> float:
    img: np.ndarray = np.array(Image.open(img_path))
    shape: Tuple[int, int, int] = img.shape

    for box in label:
        x_start, x_end, y_start, y_end = (
            int((box[1] - 0.5 * box[3]) * shape[1]),
            int((box[1] + 0.5 * box[3]) * shape[1]),
            int((box[2] - 0.5 * box[4]) * shape[0]),
            int((box[2] + 0.5 * box[4]) * shape[0]),
        )
        print(x_start, x_end, y_start, y_end)
        Image.fromarray(img[y_start:y_end, x_start:x_end]).show()

    # img[
    #     (
    #         ((img[:, :, 0] < r_val_min) | (img[:, :, 0] > r_val_max))
    #         | ((img[:, :, 1] < g_val_min) | (img[:, :, 1] > g_val_max))
    #         | ((img[:, :, 2] < b_val_min) | (img[:, :, 2] > b_val_max))
    #     )
    # ] = 0

    # Image.fromarray(img).show()


def objective(trial: optuna.trial.Trial, class_: float) -> float:
    """
    class_ should be the index of the desired class in the classes.txt file (as float)
    """
    r_min: int = trial.suggest_int("r_min", 0, 255)
    r_max: int = trial.suggest_int("r_max", 0, 255)
    g_min: int = trial.suggest_int("g_min", 0, 255)
    g_max: int = trial.suggest_int("g_max", 0, 255)
    b_min: int = trial.suggest_int("b_min", 0, 255)
    b_max: int = trial.suggest_int("b_max", 0, 255)

    path: str = "data/Labels/"
    scores: List[float] = []
    for file in os.listdir(path):
        label: List[Tuple[float, float, float, float, float]] = openLabelFile(
            path + file
        )
        label = [elem for elem in label if elem[0] == class_]
        if len(label) > 0:
            img_path: str = (
                path + file[:-3] + "png"
                if os.path.exists(path + file[:-3] + "png")
                else path + file[:-3] + "jpg"
            )
            scores.extend(
                computeScoreForImage(
                    img_path,
                    label,
                    r_min,
                    r_max,
                    g_min,
                    g_max,
                    b_min,
                    b_max,
                )
            )

    return np.average(scores)


if __name__ == "__main__":
    r_val_min = 0
    r_val_max = 40
    g_val_min = 115
    g_val_max = 165
    b_val_min = 20
    b_val_max = 130

    computeScoreForImage(
        "data/Images/0.jpg",
        openLabelFile("data/Labels/0.txt"),
        r_val_min,
        r_val_max,
        g_val_min,
        g_val_max,
        b_val_min,
        b_val_max,
    )
