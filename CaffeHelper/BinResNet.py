from caffe_user import *
import os

# ----- Configuration -----
name = "tb"
batch_size = 128
resnet_nums = [3, 4, 6, 3]  # resnet-50
fc_type = "TBInnerProduct"
conv_type = "TBConvolution"
tb_param = Parameter('tb_param')
tb_param.add_param_if('full_train', True)
tb_param.add_param_if('use_bias', False)
tb_param.add_param_if('w_binary', True)
tb_param.add_param_if('in_binary', False)
tb_param.add_param_if('clip', 0)
tb_param.add_param_if('reg', 0)
activation_method = "ReLU"
filler_xavier = Filler('xavier')
filler_constant = Filler('constant', value=0.)

other_param = [Parameter('param').add_param('lr_mult', 1).add_param('decay_mult', 1),
               Parameter('param').add_param('lr_mult', 2).add_param('decay_mult', 0),
               tb_param]

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU()
solver.test(test_iter=1000, test_interval=4000, test_initialization=False)
solver.train(base_lr=0.1, lr_policy='step', gamma=0.96, stepsize=320000, max_iter=600000, weight_decay=0.0001)
solver.optimizer(type='SGD', momentum=0.9)
solver.display(display=40, average_loss=40)
solver.snapshot(snapshot=40000, snapshot_prefix=name)

# --------- Network ----------
Net("ImageNet_" + name)
data_dir = "/home/wandiwen/data/ilsvrc12/"
data, label = \
    Data([], phase=TRAIN, source=data_dir + "ilsvrc12_train_lmdb", batch_size=batch_size, backend=Net.LMDB,
         optional_params=[Transform(mean_file=data_dir + "ilsvrc12_mean.binaryproto", crop_size=224, mirror=True)])
Data([], phase=TEST, source=data_dir + "ilsvrc12_val_lmdb", batch_size=50, backend=Net.LMDB,
     optional_params=[Transform(mean_file=data_dir + "ilsvrc12_mean.binaryproto", mirror=False, crop_size=224)])
out = [data]
label = [label]

out = Conv(out, name='resnet1_conv', num_output=64, kernel_size=7, stride=2, pad=3,
           weight_filler=filler_xavier, bias_filler=filler_constant, optional_params=other_param)
out = BN(out, name='resnet1_conv_bn')
out = Activation(out, name='resnet1_conv_bn', method=activation_method)
out = Pool(out, name='resnet1_pool', method=Net.MaxPool, kernel_size=3, stride=2)


def neck_block(out_, num_output, first=False, stride=1):
    out_ = BN(out_, name='bn')
    x = out_
    out_ = Conv(out_, name='conv1', conv_type=conv_type, num_output=num_output, kernel_size=1, stride=stride, pad=0,
                weight_filler=filler_xavier, bias_term=False, optional_params=other_param)
    out_ = BN(out_, name='conv1_bn')
    out_ = Conv(out_, name='conv2', conv_type=conv_type, num_output=num_output, kernel_size=3, stride=1, pad=1,
                weight_filler=filler_xavier, bias_term=False, optional_params=other_param)
    out_ = BN(out_, name='conv2_bn')
    out_ = Conv(out_, name='conv3', conv_type=conv_type, num_output=num_output * 4, kernel_size=1, stride=1, pad=0,
                weight_filler=filler_xavier, bias_term=False, optional_params=other_param)
    # out_ = BN(out_, name='conv3_bn')
    if first:
        x = Conv(x, name='conv_shortcut', conv_type=conv_type, num_output=num_output * 4, kernel_size=1, stride=stride,
                 pad=0, bias_term=False, weight_filler=filler_xavier, optional_params=other_param)
        # x = BN(x, name='conv_shortcut_bn')
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

out = Pool(out, name='avg_pool', method=1, global_pooling=True, stride=None, kernel_size=None)
out = FC(out, name='fc', num_output=1000, weight_filler=filler_xavier, bias_filler=filler_constant)
accuracy = Accuracy(out + label)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'resnet/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
