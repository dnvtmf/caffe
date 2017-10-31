from caffe_user import *
import os

# ----- Configuration -----
name = "Ternary"
batch_size = 100
resnet_n = 3
cifar10 = CIFAR_10(batch_size)
activation_method = "ReLU"
weight_filler = Filler('msra')
filler_bias = Filler('constant')

weight_decay = 0
t = 0.8
conv = TBBlock
tb_method = name
scale_term = True
if name == 'full':
    conv = NormalBlock
    weight_decay = 1e-4

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU(0)
solver.test(test_iter=cifar10.test_iter, test_interval=1000,
            test_initialization=False)
solver.train(base_lr=0.1, lr_policy='multistep', stepvalue=[32000, 48000],
             gamma=0.1, max_iter=64000, weight_decay=weight_decay)
solver.optimizer(type='SGD', momentum=0.9)
solver.display(display=100, average_loss=100)
solver.snapshot(snapshot=10000, snapshot_prefix='snapshot/' + name)

# --------- Network ----------
Net("cifar10_" + name)
out, label = cifar10.data()
out = NormalBlock(out, name='conv1', num_output=16, kernel_size=3, stride=1,
                  pad=1, weight_filler=weight_filler, act="ReLU")


def block(out_, num_output, first=False, stride=1):
    x = out_
    out_ = TBBlock(out_, 'A', tb_method, num_output, 3, stride, 1,
                   weight_filler, scale_term=scale_term, threshold_t=t)
    out_ = TBBlock(out_, 'B', tb_method, num_output, 3, 1, 1, weight_filler,
                   scale_term=scale_term, threshold_t=t)
    if first:
        x = TBBlock(x, 'short', tb_method, num_output, 1, stride, 0,
                    weight_filler, scale_term=scale_term, threshold_t=t)
    out_ = Eltwise(out_ + x, name='add')
    return out_


def stack(out_, num_output, number, add_stride=0):
    for i in xrange(number):
        with NameScope('branch' + str(i + 1)):
            if i == 0:
                out_ = block(out_, num_output, True, 1 + add_stride)
            else:
                out_ = block(out_, num_output)
    return out_


with NameScope('res2'):
    out = stack(out, 16, resnet_n)

with NameScope('res3'):
    out = stack(out, 32, resnet_n, add_stride=1)

with NameScope('res4'):
    out = stack(out, 64, resnet_n, add_stride=1)

out = BN(out, 'bn')
out = Activation(out, 'relu', method='ReLU')
out = Pool(out, name='avg_pool', method=Net.AveragePool, global_pooling=True,
           stride=None, kernel_size=None)
out = FC(out, name='fc', num_output=10, weight_filler=weight_filler,
         bias_filler=filler_bias)
accuracy = Accuracy(out + label)
# loss = HingeLoss(out + label, norm=2)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/resnet/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
