from caffe_user import *
import os

# ----- Configuration -----
name = "resnet"
batch_size = 128
resnet_n = 3
fc_type = "InnerProduct"
conv_type = "Convolution"
tb_param = Parameter('tb_param')
tb_param.add_param_if('full_train', True)
tb_param.add_param_if('use_bias', False)
tb_param.add_param_if('w_binary', True)
tb_param.add_param_if('in_binary', False)
tb_param.add_param_if('clip', 0)
tb_param.add_param_if('reg', 0.)
activation_method = "ReLU"
filler_xavier = Filler('xavier')
filler_constant = Filler('constant', value=0.2)

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU()
solver.test(test_iter=100, test_interval=1000, test_initialization=False)
num_iter = 64000
lr_start = 0.1
lr_end = 1e-4
lr_decay = (lr_end / lr_start) ** (1. / num_iter)
solver.train(base_lr=lr_start, lr_policy='exp',
             gamma=lr_decay, max_iter=num_iter, weight_decay=1e-4)
solver.optimizer(type='SGD', momentum=0.9)
solver.display(display=200)
solver.snapshot(snapshot=5000, snapshot_prefix=name)

# --------- Network ----------
Net("cifar10_" + name)
data, label = Data([], phase=TRAIN, source="../cifar10_train_lmdb", batch_size=batch_size, backend=Net.LMDB,
                   optional_params=[Transform(mean_file="../mean.binaryproto")])
Data([], phase=TEST, source="../cifar10_test_lmdb", batch_size=100, backend=Net.LMDB,
     optional_params=[Transform(mean_file="../mean.binaryproto")])
out = [data]
label = [label]
out = Conv(out, name='conv1', num_output=16, kernel_size=3, stride=1, pad=1,
           weight_filler=filler_xavier, bias_filler=filler_constant)
out = BN(out, name='bn_act1')
out = Activation(out, name='act1', method=activation_method)


def block(out_, num_output, first=False):
    x = out_
    out_ = BN(out_, name='bn1')
    out_ = Conv(out_, name='conv1', conv_type=conv_type, num_output=num_output, kernel_size=3, stride=1, pad=1,
                weight_filler=filler_xavier, bias_filler=filler_constant, optional_params=[tb_param])
    out_ = BN(out_, name='conv1_bn')
    out_ = Activation(out_, name='conv1_relu', method=activation_method)
    out_ = BN(out_, name='bn2')
    out_ = Conv(out_, name='conv2', conv_type=conv_type, num_output=num_output, kernel_size=3, stride=1, pad=1,
                weight_filler=filler_xavier, bias_filler=filler_constant, optional_params=[tb_param])
    out_ = BN(out_, name='conv2_bn')
    if first:
        x = BN(x, name='bn_shortcut')
        x = Conv(x, name='conv_shortcut', conv_type=conv_type, num_output=num_output, kernel_size=1, stride=1, pad=0,
                 bias_term=False, weight_filler=filler_xavier)
        x = BN(x, name='conv_shortcut_bn')
    out_ = Eltwise(out_ + x, name='add')
    out_ = Activation(out_, name='relu', method=activation_method)
    return out_


def stack(out_, num_output, number):
    for i in xrange(number):
        with NameScope('branch' + str(i)):
            out_ = block(out_, num_output, i == 0)
    return out_


with NameScope('res2'):
    out = stack(out, 16, resnet_n)

with NameScope('res3'):
    out = stack(out, 32, resnet_n)

with NameScope('res4'):
    out = stack(out, 64, resnet_n)

out = Pool(out, name='avg_pool', method=1, global_pooling=True, stride=None, kernel_size=None)
out = FC(out, name='fc', num_output=10, weight_filler=filler_xavier,
         bias_term=True, bias_filler=filler_constant)
accuracy = Accuracy(out + label)
# loss = HingeLoss(out + label, norm=2)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
