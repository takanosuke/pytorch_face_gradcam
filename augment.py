from PIL import Image
import glob
import os
import random
img_dir  = "./datasets"
for path in glob.glob(os.path.join(img_dir,"*","*")):
    img = Image.open(path)
    name, ext = os.path.splitext(path)
    if random.random() > 0.5:
        himg = img.transpose(Image.FLIP_LEFT_RIGHT)
        himg.save(name+"_flip"+ext)