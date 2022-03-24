import numpy as np
import os

from PIL import Image
from typing import Dict, List, Tuple, Set


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


class IdIterator:
    def __init__(self, class_id: int, id_list: List[str]) -> None:
        self.class_id: int = class_id
        self.id_list: List[str] = id_list
        self.label_path: str = "data/Labels/{}.txt"
        self.img_path: str = "data/Images/{}.jpg"

    def __iter__(self):
        return self

    def __next__(self):
        if self.id_list:
            id: str = self.id_list.pop()
            label: List[Tuple[float, float, float, float, float]] = openLabelFile(
                self.label_path.format(id)
            )
            if self.class_id >= 0:
                label = [elem for elem in label if elem[0] == self.class_id]
            img: np.ndarray = np.array(Image.open(self.img_path.format(id)))
            return label, img
        raise StopIteration


class DataManager:
    def __init__(self, train_split: float = 0.7) -> None:
        self.train_split: float = train_split
        folder: str = "data/Labels/"

        with open(folder + "classes.txt") as f:
            lines = f.readlines()
        self.class_mapping: Dict[int, str] = {
            i: lines[i][:-1] for i in range(len(lines))
        }
        self.classes: Dict[int, List[str]] = {i: [] for i in range(len(lines))}
        self.total_boxes: int = 0
        self.total_images: int = 0

        for file in os.listdir(folder):
            if file != "classes.txt":
                label: List[Tuple[float, float, float, float, float]] = openLabelFile(
                    folder + file
                )
                self.total_boxes += len(label)
                self.total_images += 1
                unique_labels: Set[float] = set(int(elem[0]) for elem in label)
                id: str = file[:-4]
                for unq in unique_labels:
                    self.classes[unq].append(id)

        for key in self.classes.keys():
            # sort ids to make sure this is deterministic across different systems
            self.classes[key].sort()

    def create_class_iterator(self, class_id: int, train: bool) -> IdIterator:
        temp: List[str] = self.classes[class_id]
        return (
            IdIterator(class_id, temp[: int(self.train_split * len(temp))])
            if train
            else IdIterator(class_id, temp[int(self.train_split * len(temp)) :])
        )

    def create_iterator(self) -> IdIterator:
        ids: List[List[int]] = [value for key, value in self.classes.items()]
        return IdIterator(-1, set([item for sublist in ids for item in sublist]))

    def class_distribution(self) -> Dict[int, int]:
        return {key: len(value) for key, value in self.classes.items()}


def loadImages(folder: str) -> List[np.ndarray]:
    imgs: List[np.ndarray] = []
    for file in os.listdir(folder):
        imgs.append(np.array(Image.open(folder + file)))

    return imgs
