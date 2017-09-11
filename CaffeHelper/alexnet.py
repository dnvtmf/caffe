from caffe_user import *
import os

# ----- Configuration -----
name = "tb"
batch_size = 256
fc_type = "TBInnerProduct"
conv_type = "TBConvolution"
tb_param = Parameter('tb_param')
tb_param.add_param_if('full_train', True)
tb_param.add_param_if('use_bias', True)
tb_param.add_param_if('w_binary', True)
tb_param.add_param_if('in_binary', False)
tb_param.add_param_if('clip', 3)
tb_param.add_param_if('reg', 0)
activation_method = "ReLU"
filler_xavier = Filler('xavier')
filler_uniform = Filler('uniform', min_=-0.1, max_=0.1)
filler_constant = Filler('constant', value=0.)
filler_constant2 = Filler('constant', value=0.1)
filler_gaussian = Filler('gaussian', std=0.01)
filler_gaussian2 = Filler('gaussian', std=0.005)

other_param = [Parameter('param').add_param('lr_mult', 1).add_param('decay_mult', 1),
               Parameter('param').add_param('lr_mult', 2).add_param('decay_mult', 0),
               tb_param]
# --------- Network ----------
Net("ImageNet_" + name)
data, label = Data([], phase=TRAIN, source="/home/wandiwen/data/ilsvrc12/ilsvrc12_train_lmdb", batch_size=batch_size,
                   backend=Net.LMDB, optional_params=[
        Transform(mean_file="/home/wandiwen/data/ilsvrc12/ilsvrc12_mean.binaryproto", crop_size=227, mirror=True)])
Data([], phase=TEST, source="/home/wandiwen/data/ilsvrc12/ilsvrc12_val_lmdb", batch_size=50, backend=Net.LMDB,
     optional_params=[
         Transform(mean_file="/home/wandiwen/data/ilsvrc12/ilsvrc12_mean.binaryproto", mirror=False, crop_size=227)])
out = [data]
label = [label]

out = Conv(out, name='conv1', num_output=96, bias_term=True, kernel_size=11, stride=4,
           weight_filler=filler_gaussian, bias_filler=filler_constant, optional_params=other_param)
out = Activation(out, name='act1', method=activation_method)
out = LRN(out, name='norm1', local_size=5, alpha=0.0001, beta=0.75)
out = Pool(out, name='pool1', method=Net.MaxPool, kernel_size=3, stride=2)

out = BN(out, name='bn1')
out = Conv(out, name='conv2', conv_type=conv_type, num_output=256, kernel_size=5, pad=2,
           weight_filler=filler_gaussian, bias_filler=filler_constant2, optional_params=other_param)
out = Activation(out, name='act2', method=activation_method)
out = LRN(out, name='norm2', local_size=5, alpha=0.0001, beta=0.75)
out = Pool(out, name='pool2', method=Net.MaxPool, kernel_size=3, stride=2)

out = BN(out, name='bn3')
out = Conv(out, name='conv3', conv_type=conv_type, num_output=384, kernel_size=3, pad=1,
           weight_filler=filler_gaussian, bias_filler=filler_constant, optional_params=other_param)
out = Activation(out, name='act3', method=activation_method)

out = BN(out, name='bn4')
out = Conv(out, name='conv4', conv_type=conv_type, num_output=384, kernel_size=3, pad=1, group=2,
           weight_filler=filler_gaussian, bias_filler=filler_constant2, optional_params=other_param)
out = Activation(out, name='act4', method=activation_method)

out = BN(out, name='bn5')
out = Conv(out, name='conv5', conv_type=conv_type, num_output=256, kernel_size=3, pad=1, group=2,
           weight_filler=filler_gaussian, bias_filler=filler_constant2, optional_params=other_param)
out = Activation(out, name='act5', method=activation_method)
out = Pool(out, name='pool5', method=Net.MaxPool, kernel_size=3, stride=2)

out = BN(out, name='bn6')
out = FC(out, name='conv6', fc_type=fc_type, num_output=4096, weight_filler=filler_gaussian2,
         bias_filler=filler_constant2, optional_params=other_param)
out = Activation(out, name='act6', method=activation_method)
out = DropOut(out, name='drop6', dropout_ratio=0.5)

out = BN(out, name='bn7')
out = FC(out, name='fc7', fc_type=fc_type, num_output=4096, weight_filler=filler_gaussian2,
         bias_filler=filler_constant2, optional_params=other_param)
out = Activation(out, name='act7', method=activation_method)
out = DropOut(out, name='drop7', dropout_ratio=0.5)

out = FC(out, name='fc8', num_output=1000, weight_filler=filler_gaussian, bias_filler=filler_constant)

accuracy = Accuracy(out + label)
# loss = HingeLoss(out + label, norm=2)
loss = SoftmaxWithLoss(out + label)

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU()
solver.test(test_iter=1000, test_interval=1000, test_initialization=False)
solver.train(base_lr=0.01, lr_policy='step', gamma=0.1, stepsize=100000, max_iter=450000, weight_decay=0.0005)
solver.optimizer(type='SGD', momentum=0.9)
solver.display(display=20)
solver.snapshot(snapshot=10000, snapshot_prefix=name)

model_dir = os.path.join(os.getenv('HOME'), 'alexnet/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
