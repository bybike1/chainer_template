from chainer.datasets import mnist
from chainer.datasets import split_dataset_random
from network import network
from chainer import training
from chainer import optimizers
from chainer import iterators
from chainer.training import extensions
import chainer.links as L
import numpy as np

# setup mnist dataset(example)
train_val, test = mnist.get_mnist(withlabel=True, ndim=1)
train, valid = split_dataset_random(train_val, round(len(train_val)*0.8))

# training params
batchsize = 64

# setup iterators
train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(valid, batchsize, False, False)
test_iter = iterators.SerialIterator(test, batchsize, False, False)

# gpu params
gpu_id = 0

# defining updater
net = L.Classifier(network())

# optimizer
optimizer = optimizers.SGD(lr=0.01).setup(net)

# updater
updater = training.StandardUpdater(train_iter,optimizer,device=gpu_id)

max_epoch = 10
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out = 'result')

trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(valid_iter, net, device=gpu_id), name='val')
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
trainer.extend(extensions.ParameterStatistics(net.predictor.l1, {'std': np.std}))
trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.dump_graph('main/loss'))

trainer.run()
