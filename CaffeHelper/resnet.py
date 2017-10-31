from caffe_user import *
import os

# ----- Configuration -----
name = "Ternary"
batch_size = 128
images = ImageNet(batch_size)
resnet_nums = [2, 2, 2, 2]  # resnet-18
tb_method = name
activation_method = "Sigmoid"
scale_term = True
t = 0.8
weight_filler = Filler('xavier')

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU(1)
solver.test(test_iter=images.test_iter,
            test_interval=min(1000, images.train_iter),
            test_initialization=False)
step_size = 10 * images.train_iter
max_iter = 30 * images.train_iter
solver.train(base_lr=0.1, lr_policy='step', gamma=0.1, stepsize=step_size,
             max_iter=max_iter, weight_decay=0)
solver.optimizer(type='SGD', momentum=0.9)
solver.display(display=20, average_loss=20)
solver.snapshot(snapshot=40000, snapshot_prefix='snapshot/' + name)

# --------- Network ----------
Net("ImageNet_" + name)
out, label = images.data(cs=224)

out = NormalBlock(out, 'conv1', None, 64, 7, 2, 3, weight_filler)
out = Pool(out, name='pool1', method=Net.MaxPool, kernel_size=3, stride=2)


def neck_block(out_, num_output, first=False, stride=1):
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
                out_ = neck_block(out_, num_output, True, 1 + add_stride)
            else:
                out_ = neck_block(out_, num_output)
    return out_


with NameScope('resnet2'):
    out = stack(out, 64, resnet_nums[0])

with NameScope('resnet3'):
    out = stack(out, 128, resnet_nums[1], add_stride=1)

with NameScope('resnet4'):
    out = stack(out, 256, resnet_nums[2], add_stride=1)

with NameScope('resnet5'):
    out = stack(out, 512, resnet_nums[3], add_stride=1)

out = BN(out, 'bn')
out = Activation(out, 'relu', method='ReLU')
out = Pool(out, name='avg_pool', method=1, global_pooling=True, stride=None,
           kernel_size=None)
out = FC(out, name='fc', num_output=1000, weight_filler=weight_filler)
accuracy = Accuracy(out + label)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'resnet/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
