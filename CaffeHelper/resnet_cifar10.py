from caffe_user import *
import os

# ----- Configuration -----
name = "tb"
batch_size = 128
resnet_n = 3
cifar10 = CIFAR_10(batch_size)
activation_method = "ReLU"
filler_weight = Filler('msra')
filler_bias = Filler('constant')

other_param = None
weight_decay = 0
if name == 'full':
    conv_type = "Convolution"
    weight_decay = 5e-4
elif name == 'tb':
    tb_param = Parameter('tb_param')
    tb_param.add_param_if('use_bias', False)
    tb_param.add_param_if('w_binary', True)
    tb_param.add_param_if('in_binary', False)
    tb_param.add_param_if('clip', 0)
    tb_param.add_param_if('reg', 0.)
    other_param = [tb_param]
    conv_type = "TBConvolution"
else:
    conv_type = 'XnorNetConvolution'

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU(1)
solver.test(test_iter=100, test_interval=1000, test_initialization=False)
solver.train(base_lr=0.1, lr_policy='multistep', stepvalue=[32000, 48000],
             gamma=0.1, max_iter=64000, weight_decay=weight_decay)
solver.optimizer(type='Nesterov', momentum=0.9)
solver.display(display=100, average_loss=100)
solver.snapshot(snapshot=10000, snapshot_prefix='snapshot/' + name)

# --------- Network ----------
Net("cifar10_" + name)
out, label = cifar10.data()
out = Conv(out, name='conv1', num_output=16, kernel_size=3, stride=1, pad=1,
           weight_filler=filler_weight, bias_term=True)


def block(out_, num_output, stride=1):
    x = out_
    #    out_ = BN(out_, name='bn_relu1', bias_term=True)
    #    out_ = Activation(out_, name='relu1', method=activation_method)
    out_ = BN(out_, name='bn')
    out_ = Conv(out_, name='conv1', conv_type=conv_type, num_output=num_output,
                kernel_size=3, stride=stride, pad=1,
                weight_filler=filler_weight, optional_params=other_param)

    #    out_ = BN(out_, name='bn_relu2', bias_term=True)
    #    out_ = Activation(out_, name='relu2', method=activation_method)
    out_ = BN(out_, name='bn2')
    out_ = Conv(out_, name='conv2', conv_type=conv_type, num_output=num_output,
                kernel_size=3, stride=1, pad=1,
                weight_filler=filler_weight, optional_params=other_param)

    if stride != 1:
        with NameScope('shortcut'):
            x = BN(x, name='bn', bias_term=True)
            x = Activation(x, name='act', method=activation_method)
            x = Conv(x, name='conv',
                     num_output=num_output, kernel_size=2, stride=stride,
                     pad=0, weight_filler=filler_weight,
                     optional_params=other_param)

    out_ = Eltwise(out_ + x, name='add')
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

out = BN(out, name='bn', bias_term=True)
out = Activation(out, name='relu', method=activation_method)
out = Pool(out, name='avg_pool', method=Net.AveragePool, global_pooling=True,
           stride=None, kernel_size=None)
out = FC(out, name='fc', num_output=10, weight_filler=filler_weight,
         bias_filler=filler_bias)
accuracy = Accuracy(out + label)
# loss = HingeLoss(out + label, norm=2)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/resnet/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
