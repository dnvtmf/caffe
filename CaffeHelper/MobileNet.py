from caffe_user import *
import os

# ----- Configuration -----
name = "full"
batch_size = 128
data_dir = os.path.join(os.getenv('HOME'), 'data/ilsvrc12/')
weight_filler = Filler('msra')
bias_filler = Filler('constant')
other_param = []
weight_decay = 0
if name == 'full':
    conv_type = "Convolution"
    weight_decay = 4e-5
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
solver.test(test_iter=1000, test_interval=1000, test_initialization=False)
solver.train(base_lr=0.1, lr_policy='step', gamma=0.1, stepsize=100000,
             max_iter=460000, weight_decay=weight_decay)
solver.optimizer(type='SGD', momentum=0.9)
solver.display(display=20, average_loss=20)
solver.snapshot(snapshot=10000, snapshot_prefix=name)

# --------- Network ----------
Net("ImageNet_" + name)
data, label = Data([], phase=TRAIN,
                   source=os.path.join(data_dir, "ilsvrc12_train_lmdb"),
                   batch_size=batch_size,
                   backend=Net.LMDB, optional_params=[
        Transform(
            mean_file=os.path.join(data_dir, "ilsvrc12_mean.binaryproto"),
            crop_size=224, mirror=True)])
Data([], phase=TEST, source=os.path.join(data_dir, "ilsvrc12_val_lmdb"),
     batch_size=50, backend=Net.LMDB,
     optional_params=[
         Transform(
             mean_file=os.path.join(data_dir, "ilsvrc12_mean.binaryproto"),
             mirror=False, crop_size=224)])
out = [data]
label = [label]

out = Conv(out, name='conv1', num_output=32, bias_term=False, kernel_size=3,
           stride=2, pad=1, weight_filler=weight_filler,
           optional_params=other_param)
out = BN(out, name='acn_bn1', bias_term=True, inplace=True)
out = Activation(out, name='relu1', method="ReLU", inplace=True)


def Convolution(out_, name_, num_input, num_output, kernel_size, pad, stride=1):
    with NameScope(name_):
        out_ = Conv(out_, 'dw', num_output=num_input, kernel_size=kernel_size,
                    stride=stride, pad=pad, group=num_input,
                    weight_filler=weight_filler, bias_term=False)
        out_ = BN(out_, name='dw_bn', bias_term=True, inplace=True)
        out_ = Activation(out_, name='dw_relu', method="ReLU", inplace=True)

        out_ = Conv(out_, 'conv', num_output=num_output, kernel_size=1,
                    stride=1, pad=0, weight_filler=weight_filler,
                    bias_term=False)
        out_ = BN(out_, name='conv_bn', bias_term=True, inplace=True)
        out_ = Activation(out_, name='conv_relu', method="ReLU", inplace=True)
    return out_

out = Convolution(out, 'c2', 32, 64, 3, 1, 1)
out = Convolution(out, 'c3', 64, 128, 3, 1, 2)
out = Convolution(out, 'c4', 128, 128, 3, 1, 1)
out = Convolution(out, 'c5', 128, 256, 3, 1, 2)
out = Convolution(out, 'c6', 256, 256, 3, 1, 1)
out = Convolution(out, 'c7', 256, 512, 3, 1, 2)
out = Convolution(out, 'c8_1', 512, 512, 3, 1, 1)
out = Convolution(out, 'c8_2', 512, 512, 3, 1, 1)
out = Convolution(out, 'c8_3', 512, 512, 3, 1, 1)
out = Convolution(out, 'c8_4', 512, 512, 3, 1, 1)
out = Convolution(out, 'c8_5', 512, 512, 3, 1, 1)
out = Convolution(out, 'c9', 512, 1024, 3, 1, 2)
out = Convolution(out, 'c10', 1024, 1024, 3, 1, 1)

out = Pool(out, name='avg_pool', method=Net.AveragePool, global_pooling=True,
           stride=None, kernel_size=None)
out = FC(out, name='fc', num_output=1000, weight_filler=weight_filler,
         bias_filler=bias_filler)

accuracy = Accuracy(out + label)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'MobileNet/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
