from caffe_user import *
import os

# ----- Configuration -----
name = "cnn"
num_epoch = 500
batch_size = 100
full_train = True
use_bias = True
w_binary = True
in_binary = False
activation_method = "ReLU"
filler_xavier = Filler('xavier')
filler_uniform = Filler('uniform', min_=-0.1, max_=0.1)
filler_constant = Filler('constant')

# --------- Network ----------
# 32-C5 + MP2 + 64-C5 + MP2 + 512 FC + SVM
Net("mnist_" + name)
data, label = Data([], phase=TRAIN, source="../mnist_train_lmdb", batch_size=batch_size, backend=Net.LMDB,
                   optional_params=[Transform(scale=0.00390625)])
Data([], phase=TEST, source="../mnist_test_lmdb", batch_size=100, backend=Net.LMDB,
     optional_params=[Transform(scale=0.00390625)])
out = [data]
label = [label]
out = Conv(out, name='conv1', num_output=32, bias_term=True, kernel_size=5, stride=1, pad=2,
           weight_filler=filler_xavier, bias_filler=filler_constant)
out = Activation(out, name='act1', method=activation_method)
out = Pool(out, name='pool1')
out = BN(out, name='bn1')
out = TBConv(out, name='conv2', num_output=64, bias_term=True, kernel_size=5, stride=1, pad=2,
             weight_filler=filler_xavier, bias_filler=filler_constant,
             full_train=full_train, use_bias=use_bias, w_method=w_binary, in_method=in_binary)
out = Activation(out, name='act2', method=activation_method)
out = Pool(out, name='pool2')
out = BN(out, name='bn2')
out = TBFC(out, name='fc3', num_output=512, bias_term=True, weight_filler=filler_xavier, bias_filler=filler_constant,
           full_train=full_train, use_bias=use_bias, w_method=w_binary, in_method=in_binary)
out = Activation(out, name='act3', method=activation_method)
out = FC(out, name='fc4', num_output=10, weight_filler=filler_xavier, bias_term=True, bias_filler=filler_constant)
accuracy = Accuracy(out + label)
# loss = HingeLoss(out + label, norm=2)
loss = SoftmaxWithLoss(out + label)

# ---------- solver ----
solver = Solver().net('./model.prototxt').CPU()
solver.test(test_iter=100, test_interval=500, test_initialization=False)
solver.train(base_lr=0.001, lr_policy='fixed', max_iter=3000)  # , weight_decay=1e-4)
# solver.train(base_lr=0.001, lr_policy='step', gamma=0.1, stepsize=1000, max_iter=3000)
solver.optimizer(type='SGD', momentum=0.9)
# solver.optimizer(type='Adam')
solver.display(display=100, average_loss=100)
solver.snapshot(snapshot=5000, snapshot_prefix=name)

# print solver.get_solver_proto()
# print get_prototxt()

model_dir = os.path.join(os.getenv('HOME'), 'mnist/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
