import glob
from facenet_pytorch import MTCNN
import os
from PIL import Image

image_dir = "./images"
mtcnn = MTCNN()
for i, path in enumerate(glob.glob(os.path.join(image_dir,"*.jpg"))):
    img = Image.open(path)
    img_cropped = mtcnn(img, save_path="./face_cropped_images/{}.jpg".format(str(i)))