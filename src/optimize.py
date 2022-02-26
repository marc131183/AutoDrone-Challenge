from typing import Dict
from PIL import Image
import os
import numpy as np
import json
import xmltodict as xml
import optuna
from typing import List, Dict, Tuple


def xmlToDict(path: str) -> Dict:
    with open(path, "r") as f:
        lines: str = ""
        for line in f.readlines():
            lines += line
        return json.dumps(xml.parse(lines))


def computeScoreForImage(
    path: str, r_min: int, r_max: int, g_min: int, g_max: int, b_min: int, b_max: int
) -> float:
    dic: Dict = xmlToDict(path)
    img: np.ndarray = np.array(Image.open(dic["annotation"]["path"]))


def objective(trial: optuna.trial.Trial, color: str) -> float:
    """
    color should be either "r", "g" or "b"
    """
    r_min: int = trial.suggest_int("r_min", 0, 255)
    r_max: int = trial.suggest_int("r_max", 0, 255)
    g_min: int = trial.suggest_int("g_min", 0, 255)
    g_max: int = trial.suggest_int("g_max", 0, 255)
    b_min: int = trial.suggest_int("b_min", 0, 255)
    b_max: int = trial.suggest_int("b_max", 0, 255)

    path: str = "Labels/"
    scores: List[float] = []
    for file in os.listdir(path):
        if file[2] == color:
            scores.append(computeScoreForImage(path + file))

    return np.average(scores)


print(xmlToDict("Labels/0_g.xml"))

imgs = []

path = "data_custom/"
for file in os.listdir(path):
    imgs.append(np.array(Image.open(path + file)))

temp = np.array(Image.open(path + "0_g.jpg"))  # imgs[0].copy()
optimal_green = np.average(np.average(temp[300:340, 150:220], axis=0), axis=0)


r_val_min = 0
r_val_max = 40
g_val_min = 115
g_val_max = 165
b_val_min = 20
b_val_max = 130

temp[
    (
        ((temp[:, :, 0] < r_val_min) | (temp[:, :, 0] > r_val_max))
        | ((temp[:, :, 1] < g_val_min) | (temp[:, :, 1] > g_val_max))
        | ((temp[:, :, 2] < b_val_min) | (temp[:, :, 2] > b_val_max))
    )
] = 0

Image.fromarray(temp).show()
