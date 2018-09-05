from chainer.datasets import split_dataset_random
from network import network
from chainer import training
from chainer import optimizers
from chainer import iterators
from chainer.training import extensions
import chainer.links as L
import cupy as xp
from chainer.datasets import tuple_dataset
import cv2
import chainer
from chainer import cuda, serializers, Variable
import numpy as np
from PIL import Image
import sys
import os

def load_data(path, crop=True, size=None, mode="label", xp=np):
    img = Image.open(path)
    if crop:
        w,h = img.size
        if w < h:
            if w < size:
                img = img.resize((size, size*h//w))
                w, h = img.size
        else:
            if h < size:
                img = img.resize((size*w//h, size))
                w, h = img.size
        img = img.crop((int((w-size)*0.5), int((h-size)*0.5), int((w+size)*0.5), int((h+size)*0.5)))

    if mode=="label":
        y = xp.asarray(img, dtype=xp.int32)
        mask = y == 255
        #mask = mask.astype(xp.int32)
        y[mask] = -1
        return y

    elif mode=="data":
        x = xp.asarray(img, dtype=xp.float32).transpose(2, 0, 1)
        #x -= 120
        return x

    elif mode=="predict":
        return img
def get_bit(byte_val, idx):
    return int((byte_val & (1 << idx)) != 0)

def shift_bit(byte_val, idx):
    return byte_val << idx if idx >= 0 else byte_val >> (-idx)

def bitor(a, b):
    return a | b
def make_color_map():
    n = 256
    cmap = np.zeros((n, 3)).astype(np.int32)
    for i in range(0, n):
        d = i - 1
        r,g,b = 0,0,0
        for j in range(0, 7):
            r = bitor(r, shift_bit(get_bit(d, 0), 7 - j))
            g = bitor(g, shift_bit(get_bit(d, 1), 7 - j))
            b = bitor(b, shift_bit(get_bit(d, 2), 7 - j))
            d = shift_bit(d, -3)
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap[1:22]


net = network(n_class=21)

weight_file = os.path.join('result','snapshot_epoch-1')
serializers.load_npz(weight_file,net,strict=False)

inputfile = "JPEGImages/2007_000033.jpg"

img = load_data(inputfile, crop=True, size = 256,mode = "data")
img = np.expand_dims(img,axis=0)
pred = net(img).data
pred = pred[0].argmax(axis=0)
print(pred)
print(img)
cv2.imwrite("a.png",pred)

color_map = make_color_map()
row, col = pred.shape
dst = np.ones((row,col,3))
for i in range(21):
    dst[pred == i] = color_map[i]
tmp = Image.fromarray(np.uint8(dst))

b,g,r = tmp.split()
tmp = Image.merge("RGB", (r,g,b))
tmp.save('tmp.png','png')
trans = Image.new('RGBA', tmp.size, (0,0,0,0))
w, h = tmp.size
for x in range(w):
    for y in range(h):
        pixel = tmp.getpixel((x, y))
        if (pixel[0] == 0   and pixel[1] == 0   and pixel[2] == 0)or \
           (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
            continue
        trans.putpixel((x, y), pixel)

trans.save("pred.png")
