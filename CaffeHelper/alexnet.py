from caffe_user import *
import os

# ----- Configuration -----
name = "XnorNet"
batch_size = 128
images = ImageNet(batch_size)
activation_method = "ReLU"
weight_filler = Filler('msra')
bias_filler = Filler('constant')

other_param = []
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
    other_param = other_param + [tb_param]
    conv_type = "TBConvolution"
else:
    conv_type = 'XnorNetConvolution'

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU()
solver.test(test_iter=images.num_test / batch_size, test_interval=1000,
            test_initialization=False)
solver.train(base_lr=0.1, lr_policy='step', gamma=0.1, stepsize=100000,
             max_iter=460000, weight_decay=weight_decay)
solver.optimizer(type='SGD', momentum=0.9)
solver.display(display=20, average_loss=20)
solver.snapshot(snapshot=10000, snapshot_prefix=name)

# --------- Network ----------
Net("ImageNet_" + name)
out, label = images.data(cs=227)

out = Conv(out, name='conv1', num_output=96, bias_term=True, kernel_size=11,
           stride=4, weight_filler=weight_filler, bias_filler=bias_filler,
           optional_params=None)
out = BN(out, name='acn_bn1', eps=1e-5)
out = Activation(out, name='act1', method=activation_method)
out = Pool(out, name='pool1', method=Net.MaxPool, kernel_size=3, stride=2)


def Convolution(out_, cname, num_output, kernel_size, pad):
    with NameScope(cname):
        if name != 'full':
            out_ = BN(out_, name='bn', eps=1e-4)
        out_ = Conv(out_, name='conv', conv_type=conv_type,
                    num_output=num_output,
                    kernel_size=kernel_size, pad=pad,
                    weight_filler=weight_filler,
                    bias_filler=bias_filler,
                    optional_params=other_param)
        if name == 'full':
            out_ = BN(out_, name='act_bn', eps=1e-3)
            out_ = Activation(out_, name='act', method=activation_method)
    return out_


out = Convolution(out, 'conv2', 256, 5, 2)
out = Pool(out, name='pool2', method=Net.MaxPool, kernel_size=3, stride=2)

out = Convolution(out, 'conv3', 384, 3, 1)
out = Convolution(out, 'conv4', 384, 3, 1)
out = Convolution(out, 'conv5', 256, 3, 1)
out = Pool(out, name='pool5', method=Net.MaxPool, kernel_size=3, stride=2)

out = Convolution(out, 'conv6', 4096, 6, 0)
out = Convolution(out, 'conv7', 4096, 1, 0)

if name != 'full':
    out = BN(out, name='bn8', eps=1e-3)
    out = Activation(out, name='relu8', method=activation_method)
out = FC(out, name='fc8', num_output=1000, weight_filler=weight_filler,
         bias_filler=bias_filler)

accuracy = Accuracy(out + label)
# loss = HingeLoss(out + label, norm=2)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'alexnet/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
