from caffe_user import *
import os

# ----- Configuration -----
name = "tb"
num_epoch = 500
batch_size = 50
cifar10 = CIFAR_10(batch_size, False)
weight_filler = Filler('msra')
filler_constant = Filler('constant')

other_param = None
weight_decay = 0
if name == 'full':
    conv_type = "Convolution"
    weight_decay = 0.004
elif name == 'tb':
    tb_param = Parameter('tb_param')
    tb_param.add_param_if('use_bias', False)
    tb_param.add_param_if('w_binary', True)
    tb_param.add_param_if('in_binary', False)
    tb_param.add_param_if('clip', 0)
    tb_param.add_param_if('reg', 0)
    other_param = [tb_param]
    conv_type = "TBConvolution"
else:
    conv_type = 'XnorNetConvolution'

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU(1)
solver.test(test_iter=cifar10.test_iter, test_interval=cifar10.train_iter,
            test_initialization=False)
num_iter = num_epoch * cifar10.train_iter
lr_start = 0.001
lr_end = 1e-5
lr_decay = (lr_start - lr_end) ** (1. / num_iter)
solver.train(base_lr=lr_start, lr_policy='exp', gamma=lr_decay,
             max_iter=num_iter,
             weight_decay=weight_decay)
solver.optimizer(type='Adam')
solver.display(display=200, average_loss=200)
solver.snapshot(snapshot=10000, snapshot_prefix='snapshot/' + name)

# --------- Network ----------
# 2x(128C3)-MP2-2x(256C3)-MP2-2x(512C3)-MP2-2x(1024FC)-SVM
Net("cifar10_" + name)
out, label = cifar10.data()
out = Conv(out, name='conv1', num_output=128, bias_term=True, kernel_size=3,
           stride=1, pad=1, weight_filler=weight_filler)
if name == 'full':
    out = BN(out, name='bn_act1', bias_term=True, inplace=True)
    out = Activation(out, name='act1', method="ReLU", inplace=True)


def Convolution(out_, name_, num_output, kernel_size, stride, pad):
    with NameScope(name_):
        if name != 'full':
            out_ = BN(out_, name='bn')
        out_ = Conv(out_, conv_type=conv_type, num_output=num_output,
                    kernel_size=kernel_size, stride=stride, pad=pad,
                    weight_filler=weight_filler, bias_term=True,
                    optional_params=other_param)
        if name == 'full':
            out_ = BN(out_, name='conv_bn', bias_term=True, inplace=True)
            out_ = Activation(out_, name='relu', method="ReLU", inplace=True)
    return out_


out = Convolution(out, 'conv2', 128, 3, 1, 1)
out = Pool(out, name='pool1')

out = Convolution(out, 'conv3', 256, 3, 1, 1)
out = Convolution(out, 'conv4', 256, 3, 1, 1)
out = Pool(out, name='pool2')

out = Convolution(out, 'conv5', 512, 3, 1, 1)
out = Convolution(out, 'conv6', 512, 3, 1, 1)
out = Pool(out, name='pool3')

out = Convolution(out, 'fc1', 1024, 4, 1, 0)
out = Convolution(out, 'fc2', 1024, 1, 1, 0)

if name != 'full':
    out = BN(out, name='fc_bn', bias_term=True)
    out = Activation(out, name='fc_relu', method="ReLU")

out = FC(out, name='fc3', num_output=10, weight_filler=weight_filler,
         bias_term=True, bias_filler=filler_constant)
accuracy = Accuracy(out + label)
loss = HingeLoss(out + label, norm=2)

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/cnn/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
