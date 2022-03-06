import os
import numpy as np
import optuna
import pickle

from PIL import Image
from typing import List, Dict, Tuple
from boxes import getBoxes


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

    labeled: np.ndarray = np.zeros((shape[0], shape[1]), dtype=bool)
    estimated: np.ndarray = labeled.copy()

    for box in label:
        x_start, x_end, y_start, y_end = (
            int((box[1] - 0.5 * box[3]) * shape[1]),
            int((box[1] + 0.5 * box[3]) * shape[1]),
            int((box[2] - 0.5 * box[4]) * shape[0]),
            int((box[2] + 0.5 * box[4]) * shape[0]),
        )

        labeled[y_start:y_end, x_start:x_end] = True

    boxes = getBoxes(img, r_min, r_max, g_min, g_max, b_min, b_max)
    for min_row, min_col, max_row, max_col in boxes:
        estimated[min_row:max_row, min_col:max_col] = True

    # compute pixelwise difference between labeled and estimated and scale it between 0 and 1
    return np.sum(labeled ^ estimated) / (shape[0] * shape[1])


def objective(trial: optuna.trial.Trial, class_: float) -> float:
    """
    class_ should be the index of the desired class in the classes.txt file (as float)
    """
    r_min: int = trial.suggest_int("r_min", 0, 255)
    r_max: int = trial.suggest_int("r_max", r_min, 255)
    g_min: int = trial.suggest_int("g_min", 0, 255)
    g_max: int = trial.suggest_int("g_max", g_min, 255)
    b_min: int = trial.suggest_int("b_min", 0, 255)
    b_max: int = trial.suggest_int("b_max", b_min, 255)

    path: str = "data/Labels/"
    scores: List[float] = []
    for file in os.listdir(path):
        if file != "classes.txt":
            label: List[Tuple[float, float, float, float, float]] = openLabelFile(
                path + file
            )
            label = [elem for elem in label if elem[0] == class_]
            if len(label) > 0:
                img_path: str = (
                    "data/Images/" + file[:-3] + "png"
                    if os.path.exists("data/Images/" + file[:-3] + "png")
                    else "data/Images/" + file[:-3] + "jpg"
                )
                scores.append(
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


def saveStudy(study: optuna.study.Study, path: str) -> None:
    with open(path, "wb") as output:
        pickle.dump(study, output, pickle.HIGHEST_PROTOCOL)


def loadStudy(path: str) -> optuna.study.Study:
    with open(path, "rb") as input:
        return pickle.load(input)


if __name__ == "__main__":
    # Possible improvements:
    # - include max_island_size parameter in study? change it to % of image?
    # - for the objective function should we aim for only capturing the buoys
    #   and ignore any other extra boxes? (to make sure we capture all buoys)
    color_to_optimize: str = "yellow"
    mapping: Dict[str, int] = {"red": 0, "green": 1, "yellow": 2}
    study_path = "data/Studies/study_{}.pkl".format(color_to_optimize)

    # 0: study, 1: test
    mode = 1

    if mode == 0:
        """Study"""
        if not os.path.exists(study_path):
            study: optuna.study.Study = optuna.create_study()
        else:
            study: optuna.study.Study = loadStudy(study_path)

        study.optimize(lambda x: objective(x, mapping[color_to_optimize]), n_trials=500)
        saveStudy(study, study_path)

        print(study.best_params)
    else:
        """Test"""
        study: optuna.study.Study = loadStudy(study_path)

        r_min = study.best_params["r_min"]
        r_max = study.best_params["r_max"]
        g_min = study.best_params["g_min"]
        g_max = study.best_params["g_max"]
        b_min = study.best_params["b_min"]
        b_max = study.best_params["b_max"]

        img: np.ndarray = np.array(Image.open("data/Images/3.jpg"))

        boxes = getBoxes(
            img, r_min, r_max, g_min, g_max, b_min, b_max, min_island_size=50
        )
        for min_row, min_col, max_row, max_col in boxes:
            img[min_row:max_row, min_col - 1 : min_col + 2] = 0
            img[min_row:max_row, max_col - 1 : max_col + 2] = 0
            img[min_row - 1 : min_row + 2, min_col:max_col] = 0
            img[max_row - 1 : max_row + 2, min_col:max_col] = 0

        Image.fromarray(img).show()
