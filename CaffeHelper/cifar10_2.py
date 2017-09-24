from caffe_user import *
import os

# ----- Configuration -----
name = "cnn2"
num_epoch = 50
batch_size = 50
fc_type = "TBInnerProduct"
conv_type = "TBConvolution"
tb_param = Parameter('tb_param')
tb_param.add_param_if('use_bias', False)
tb_param.add_param_if('w_binary', True)
tb_param.add_param_if('in_binary', True)
tb_param.add_param_if('clip', 0)
tb_param.add_param_if('reg', 0.)
activation_method = "Sigmoid"
filler_xavier = Filler('xavier')
filler_constant = Filler('constant')

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU()
solver.test(test_iter=100, test_interval=1000, test_initialization=False)
num_iter = num_epoch * 50000 / batch_size
solver.train(base_lr=0.1, lr_policy='step', stepsize=num_iter / 5, gamma=0.1, max_iter=num_iter, weight_decay=1e-4)
solver.optimizer(type='SGD')
solver.display(display=200, average_loss=200)
solver.snapshot(snapshot=5000, snapshot_prefix=name)

# --------- Network ----------
# 32C5 - sigmoid - MaxPool3 - 32C5 - sigmoid - MaxPool3 - 64C5 - sigmoid - AvePool3 - 10FC - Softmax
Net("cifar10_" + name)
data, label = Data([], phase=TRAIN, source="../cifar10_train_lmdb", batch_size=batch_size, backend=Net.LMDB,
                   optional_params=[Transform(mean_file="../mean.binaryproto")])
Data([], phase=TEST, source="../cifar10_test_lmdb", batch_size=batch_size, backend=Net.LMDB,
     optional_params=[Transform(mean_file="../mean.binaryproto")])
out = [data]
label = [label]

out = BN(out, name='bn1')
out = Conv(out, name='conv1', num_output=32, conv_type=conv_type, bias_term=False, kernel_size=3, stride=1, pad=1,
           weight_filler=filler_xavier)
out = BN(out, name='bn_act1')
out = Activation(out, name='act1', method=activation_method)
out = Pool(out, name='pool1', method=Net.MaxPool, kernel_size=3)

out = BN(out, name='bn2')
out = Conv(out, name='conv2', conv_type=conv_type, num_output=32, bias_term=False, kernel_size=3, stride=1, pad=1,
           weight_filler=filler_xavier, optional_params=[tb_param])
out = BN(out, name='bn_act2')
out = Activation(out, name='act2', method=activation_method)
out = Pool(out, name='pool2', method=Net.MaxPool, kernel_size=3)

out = BN(out, name='bn3')
out = Conv(out, name='conv3', conv_type=conv_type, num_output=64, bias_term=False, kernel_size=3, stride=1, pad=1,
           weight_filler=filler_xavier, optional_params=[tb_param])
out = BN(out, name='bn_act3')
out = Activation(out, name='act3', method=activation_method)
out = Pool(out, name='pool3', method=Net.AveragePool, stride=3)

out = BN(out, name='bn_fc')
out = FC(out, name='fc', num_output=10, fc_type=fc_type, weight_filler=filler_xavier, bias_term=True,
         bias_filler=filler_constant)
accuracy = Accuracy(out + label)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
