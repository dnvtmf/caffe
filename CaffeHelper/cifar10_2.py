from caffe_user import *
import os

# ----- Configuration -----
name = "cnn_tb"
num_epoch = 500
batch_size = 50
fc_type = "TBInnerProduct"
conv_type = "TBConvolution"
tb_param = Parameter('tb_param')
tb_param.add_param_if('full_train', True)
tb_param.add_param_if('use_bias', True)
tb_param.add_param_if('w_binary', True)
tb_param.add_param_if('in_binary', False)
tb_param.add_param_if('clip', 3)
tb_param.add_param_if('reg', 0.)
activation_method = "ReLU"
filler_xavier = Filler('xavier')
filler_uniform = Filler('uniform', min_=-0.1, max_=0.1)
filler_constant = Filler('constant')

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU()
solver.test(test_iter=100, test_interval=1000, test_initialization=False)
num_iter = num_epoch * 50000 / batch_size
lr_start = 0.003
lr_end = 0.000002
lr_decay = (lr_end / lr_start)**(1. / num_iter)
solver.train(base_lr=lr_start, lr_policy='exp',
             gamma=lr_decay, max_iter=num_iter, weight_decay=0)
solver.optimizer(type='Adam')
solver.display(display=200)
solver.snapshot(snapshot=5000, snapshot_prefix=name)

# --------- Network ----------
# 2x(128C3)-MP2-2x(256C3)-MP2-2x(512C3)-MP2-2x(1024FC)-SVM
Net("cifar10_" + name)
data, label = Data([], phase=TRAIN, source="../cifar10_train_lmdb", batch_size=batch_size, backend=Net.LMDB,
                   optional_params=[Transform(mean_file="../mean.binaryproto")])
Data([], phase=TEST, source="../cifar10_test_lmdb", batch_size=100, backend=Net.LMDB,
     optional_params=[Transform(mean_file="../mean.binaryproto")])
out = [data]
label = [label]
out = Conv(out, name='conv1', num_output=128, bias_term=True, kernel_size=3, stride=1, pad=1,
           weight_filler=filler_xavier, bias_filler=filler_constant)
out = BN(out, name='bn_act1')
out = Activation(out, name='act1', method=activation_method)
out = BN(out, name='bn1')
out = Conv(out, name='conv2', conv_type=conv_type, num_output=128, bias_term=True, kernel_size=3, stride=1, pad=1,
           weight_filler=filler_xavier, bias_filler=filler_constant, optional_params=[tb_param])
out = BN(out, name='bn_act2')
out = Activation(out, name='act2', method=activation_method)
out = Pool(out, name='pool1')

out = BN(out, name='bn3')
out = Conv(out, name='conv3', conv_type=conv_type, num_output=256, bias_term=True, kernel_size=3, stride=1, pad=1,
           weight_filler=filler_xavier, bias_filler=filler_constant, optional_params=[tb_param])
out = BN(out, name='bn_act3')
out = Activation(out, name='act3', method=activation_method)
out = BN(out, name='bn4')
out = Conv(out, name='conv4', conv_type=conv_type, num_output=256, bias_term=True, kernel_size=3, stride=1, pad=1,
           weight_filler=filler_xavier, bias_filler=filler_constant, optional_params=[tb_param])
out = BN(out, name='bn_act4')
out = Activation(out, name='act4', method=activation_method)
out = Pool(out, name='pool2')

out = BN(out, name='bn5')
out = Conv(out, name='conv5', conv_type=conv_type, num_output=512, bias_term=True, kernel_size=3, stride=1, pad=1,
           weight_filler=filler_xavier, bias_filler=filler_constant, optional_params=[tb_param])
out = BN(out, name='bn_act5')
out = Activation(out, name='act5', method=activation_method)
out = BN(out, name='bn6')
out = Conv(out, name='conv6', conv_type=conv_type, num_output=512, bias_term=True, kernel_size=3, stride=1, pad=1,
           weight_filler=filler_xavier, bias_filler=filler_constant, optional_params=[tb_param])
out = BN(out, name='bn_act6')
out = Activation(out, name='act6', method=activation_method)
out = Pool(out, name='pool3')

out = BN(out, name='bn7')
out = FC(out, name='fc7', fc_type=fc_type, num_output=1024, bias_term=True, weight_filler=filler_xavier,
         bias_filler=filler_constant, optional_params=[tb_param])
out = BN(out, name='bn_act7')
out = Activation(out, name='act7', method=activation_method)
out = BN(out, name='bn8')
out = FC(out, name='fc8', fc_type=fc_type, num_output=1024, bias_term=True, weight_filler=filler_xavier,
         bias_filler=filler_constant, optional_params=[tb_param])
out = BN(out, name='bn_act8')
out = Activation(out, name='act8', method=activation_method)

out = FC(out, name='fc9', num_output=10, weight_filler=filler_xavier,
         bias_term=True, bias_filler=filler_constant)
accuracy = Accuracy(out + label)
loss = HingeLoss(out + label, norm=2)

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
