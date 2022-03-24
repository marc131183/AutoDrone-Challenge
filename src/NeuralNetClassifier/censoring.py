import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from typing import Tuple

from src.functions import DataManager


def createCensoringDataset(
    resize_img: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    dm = DataManager()
    x = np.empty((dm.total_boxes + dm.total_images, *resize_img, 3), dtype=np.uint8)
    y = np.empty((dm.total_boxes + dm.total_images, 5))
    index: int = 0

    for label, img in dm.create_iterator():
        img = np.array(Image.fromarray(img).resize(resize_img))

        for box in label:
            x[index] = img
            y[index] = np.array([True, *box[1:]])

            x_start, x_end, y_start, y_end = (
                int((box[1] - 0.5 * box[3]) * resize_img[1]),
                int((box[1] + 0.5 * box[3]) * resize_img[1]),
                int((box[2] - 0.5 * box[4]) * resize_img[0]),
                int((box[2] + 0.5 * box[4]) * resize_img[0]),
            )
            img[y_start:y_end, x_start:x_end] = 0
            index += 1

        x[index] = img
        y[index] = np.array([False, *np.random.random_sample((4,))])
        index += 1

    return x, y


if __name__ == "__main__":
    createCensoringDataset()
