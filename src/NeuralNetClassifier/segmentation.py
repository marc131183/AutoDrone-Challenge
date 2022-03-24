import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from PIL import Image

from src.functions import DataManager


def createSegmentationDataset(
    num_splits: Tuple[float, float],
    resize_img: Tuple[int, int] = (320, 320),
    train_split: float = 0.8,
    binary_y: bool = True,
):
    dm: DataManager = DataManager(train_split)
    x, y = [], []

    for class_ in dm.class_mapping.keys():
        for train in [True, False]:
            for label, img in dm.create_class_iterator(class_, train):
                img = np.array(Image.fromarray(img).resize(resize_img))
                for i in range(1, num_splits[0]):
                    img[int(i / num_splits[0] * resize_img[0])] = 0
                for i in range(1, num_splits[1]):
                    img[:, int(i / num_splits[1] * resize_img[1])] = 0

                plt.imshow(img)
                plt.draw()
                plt.pause(2)


if __name__ == "__main__":
    createSegmentationDataset(
        num_splits=(5, 5), resize_img=(320, 320), train_split=0.8, binary_y=True
    )
