from PIL import Image
import os

if __name__ == "__main__":
    folder = "data/Images/images_new/"
    offset = 34

    for i, file in enumerate(os.listdir(folder)):
        img = Image.open(folder + file)
        img.save("data/Images/{}.jpg".format(i + offset))
