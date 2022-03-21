import numpy as np
import os

from PIL import Image
from typing import List, Tuple


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


class ImageLabelIterator:
    def __iter__(self) -> None:
        self.label_path: str = "data/Labels/{}.txt"
        self.img_path: str = "data/Images/{}.jpg"
        self.paths: List[str] = [
            (file[:-4]) for file in os.listdir("data/Labels/") if file != "classes.txt"
        ]
        return self

    def __next__(
        self,
    ) -> Tuple[List[Tuple[float, float, float, float, float]], np.ndarray]:
        if self.paths:
            path = self.paths.pop()
            label: List[Tuple[float, float, float, float, float]] = openLabelFile(
                self.label_path.format(path)
            )
            img: np.ndarray = np.array(Image.open(self.img_path.format(path)))
            return label, img
        raise StopIteration
