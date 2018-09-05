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
# setup mnist dataset(example)
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

JpegPath = "JPEGImages/"
SegPath = "SegmentationClass/"
batchsize = 10
classes = 21
max_epoch = 10000
with open("train.txt","r") as f:
    ls = f.readlines()
names = [l.strip('\n') for l in ls]
n_data = len(names)
n_iter = n_data // batchsize
train_val = tuple_dataset.TupleDataset([load_data(JpegPath + names[i] + ".jpg", crop=True,size = 256, mode = "data",xp = np) for i in range(n_data)],
                [load_data(SegPath + names[i] + ".png", crop=True,size = 256, mode = "label",xp = np) for i in range(n_data)])

#test = tuple_dataset.TupleDataset([load_data(SegPath + names[i] + ".png", crop=True,size = 256, mode = "label",xp = xp) for i in range(n_data)])

train, valid = split_dataset_random(train_val, round(n_iter * 0.8))


# setup iterators
train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(valid, batchsize, False, False)

# gpu params
gpu_id = 0

# defining updater
net = L.Classifier(network(classes))
net.to_gpu()

# optimizer
optimizer = optimizers.Adam(alpha=0.01)
optimizer.setup(net)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_fcn')

# updater
updater = training.StandardUpdater(train_iter, optimizer, device = gpu_id)
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out = 'result')

"""
for epoch in range(1,max_epoch + 1):
    print(epoch)
    for i in range(n_iter):
        net.zerograds()
        indices = range(i * batchsize, (i+1)*batchsize)

        x = xp.zeros((batchsize, 3, 256, 256),dtype = np.float32)
        y = xp.zeros((batchsize, 256, 256),dtype = np.int32)
        for j in range(batchsize):
            name = names[i*batchsize + j]
            x[j] = load_data(JpegPath + name + ".jpg", crop=True,size = 256, mode = "data",xp = xp)
            y[j] = load_data(SegPath + name + ".png", crop=True,size = 256, mode = "label",xp = xp)
            #label = cv2.imread(SegPath + files + ".png")
        x = Variable(x)
        y = Variable(y)
        loss = net(x, t=y, train=True)

        sys.stdout.write("\r%s" % "batch: {}/{}, loss: {}".format(i+1, n_iter, loss.data))
        sys.stdout.flush()

        loss.backward()
        optimizer.update()
    print("\n"+"-"*40)
    serializers.save_npz(str(epoch) + '.weight',net)
    serializers.save_npz(str(epoch) + '.state',optimizer)
serializers.save_npz('final.weight',net)
serializers.save_npz('final.state',optimizer)
"""

trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(valid_iter, net, device=gpu_id), name='val')
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.dump_graph('main/loss'))

#print(names)
trainer.run()
