from caffe_user import *
import os

# ----- Configuration -----
name = "resnet_XnorNet"
batch_size = 128
resnet_n = 3
fc_type = "TBInnerProduct"
conv_type = "TBConvolution"
tb_param = Parameter('tb_param')
tb_param.add_param_if('full_train', True)
tb_param.add_param_if('use_bias', False)
tb_param.add_param_if('w_binary', True)
tb_param.add_param_if('in_binary', True)
tb_param.add_param_if('clip', 0)
tb_param.add_param_if('reg', 0.)
activation_method = "ReLU"
filler_xavier = Filler('xavier')
filler_constant = Filler('constant')

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU()
solver.test(test_iter=100, test_interval=1000, test_initialization=True)
solver.train(base_lr=0.1, lr_policy='multistep', stepvalue=[32000, 48000],
             gamma=0.1, max_iter=64000, weight_decay=1e-4)
solver.optimizer(type='SGD', momentum=0.9)
solver.display(display=200, average_loss=200)
solver.snapshot(snapshot=10000, snapshot_prefix=name)

# --------- Network ----------
Net("cifar10_" + name)
data, label = Data([], phase=TRAIN, source="../cifar10_train_lmdb",
                   batch_size=batch_size, backend=Net.LMDB,
                   optional_params=[
                       Transform(mean_file="../mean.binaryproto", mirror=True)])
Data([], phase=TEST, source="../cifar10_test_lmdb", batch_size=100,
     backend=Net.LMDB,
     optional_params=[Transform(mean_file="../mean.binaryproto")])
out = [data]
label = [label]

out = Conv(out, name='conv1', num_output=16, kernel_size=3, stride=1, pad=1,
           weight_filler=filler_xavier)
out = BN(out, name='bn_act1')
out = Activation(out, name='act1', method=activation_method)


def block(out_, num_output, stride=1):
    out_ = BN(out_, name='bn1')
    x = out_
    out_ = Conv(out_, name='conv1', conv_type=conv_type, num_output=num_output,
                kernel_size=3, stride=stride, pad=1,
                weight_filler=filler_xavier, optional_params=[tb_param])
    out_ = BN(out_, name='conv1_bn')
    out_ = Activation(out_, name='conv1_relu', method=activation_method)
    out_ = BN(out_, name='bn2')
    out_ = Conv(out_, name='conv2', conv_type=conv_type, num_output=num_output,
                kernel_size=3, stride=1, pad=1,
                weight_filler=filler_xavier, optional_params=[tb_param])
    out_ = BN(out_, name='conv2_bn')
    if stride != 1:
        x = Conv(x, name='conv_shortcut', conv_type=conv_type,
                 num_output=num_output, kernel_size=1, stride=stride,
                 pad=0, weight_filler=filler_xavier, optional_params=[tb_param])
        x = BN(x, name='conv_shortcut_bn')
    out_ = Eltwise(out_ + x, name='add')
    out_ = Activation(out_, name='relu', method=activation_method)
    return out_


def stack(out_, num_output, number, add_stride=0):
    for i in xrange(number):
        with NameScope('branch' + str(i + 1)):
            if i == 0:
                out_ = block(out_, num_output, 1 + add_stride)
            else:
                out_ = block(out_, num_output)
    return out_


with NameScope('res2'):
    out = stack(out, 16, resnet_n)

with NameScope('res3'):
    out = stack(out, 32, resnet_n, add_stride=1)

with NameScope('res4'):
    out = stack(out, 64, resnet_n, add_stride=1)

out = Pool(out, name='avg_pool', method=1, global_pooling=True, stride=None,
           kernel_size=None)
out = FC(out, name='fc', num_output=10, weight_filler=filler_xavier,
         bias_filler=filler_constant)
accuracy = Accuracy(out + label)
# loss = HingeLoss(out + label, norm=2)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
