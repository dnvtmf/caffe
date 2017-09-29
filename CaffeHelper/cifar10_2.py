from caffe_user import *
import os

# ----- Configuration -----
name = "tb"
num_epoch = 60
batch_size = 50
activation_method = "Sigmoid"
weight_filler = Filler('xavier')
filler_constant = Filler('constant')

other_param = None
if name == 'full':
    conv_type = "Convolution"
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
solver = Solver().net('./model.prototxt').GPU()
solver.test(test_iter=100, test_interval=1000, test_initialization=False)
num_iter = num_epoch * 50000 / batch_size
solver.train(base_lr=0.01, lr_policy='step', stepsize=num_iter / 3, gamma=0.1,
             max_iter=num_iter, weight_decay=None)
solver.optimizer(type='Adam')
solver.display(display=200, average_loss=200)
solver.snapshot(snapshot=5000, snapshot_prefix=name)

# --------- Network ----------
# 32C5 - sigmoid - MaxPool3 - LRN - 32C5 - sigmoid - MaxPool3 - LRN
#  - 64C5 - sigmoid - AvePool3 - 10FC - Softmax
Net("cifar10_" + name)
data, label = Data([], phase=TRAIN, source="../../cifar10_train_lmdb",
                   batch_size=batch_size, backend=Net.LMDB,
                   optional_params=[
                       Transform(mean_file="../../mean.binaryproto")])
Data([], phase=TEST, source="../../cifar10_test_lmdb", batch_size=batch_size,
     backend=Net.LMDB,
     optional_params=[Transform(mean_file="../../mean.binaryproto")])
out = [data]
label = [label]

out = BN(out, name='bn1')
out = Conv(out, name='conv1', num_output=32, bias_term=False, kernel_size=3,
           stride=1, pad=1,
           weight_filler=weight_filler)
out = BN(out, name='bn_act1', bias_term=True)
out = Activation(out, name='act1', method=activation_method)
out = Pool(out, name='pool1', method=Net.MaxPool, kernel_size=3)
out = LRN(out, name='lrn1')

out = BN(out, name='bn2')
out = Conv(out, name='conv2', conv_type=conv_type, num_output=32,
           bias_term=False, kernel_size=3, stride=1, pad=1,
           weight_filler=weight_filler, optional_params=other_param)
out = BN(out, name='bn_act2', bias_term=True)
out = Activation(out, name='act2', method=activation_method)
out = Pool(out, name='pool2', method=Net.MaxPool, kernel_size=3)
out = LRN(out, name='lrn2')

out = BN(out, name='bn3')
out = Conv(out, name='conv3', conv_type=conv_type, num_output=64,
           bias_term=False, kernel_size=3, stride=1, pad=1,
           weight_filler=weight_filler, optional_params=other_param)
out = BN(out, name='bn_act3', bias_term=True)
out = Activation(out, name='act3', method=activation_method)
out = Pool(out, name='pool3', method=Net.AveragePool, kernel_size=3)

out = BN(out, name='bn_fc')
out = FC(out, name='fc', num_output=10, weight_filler=weight_filler,
         bias_term=True, bias_filler=filler_constant)
accuracy = Accuracy(out + label)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/cnn2/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
