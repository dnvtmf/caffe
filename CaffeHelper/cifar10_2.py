from caffe_user import *
import os

# ----- Configuration -----
name = "tb"
num_epoch = 60
batch_size = 100
cifar10 = CIFAR_10(batch_size)
activation_method = "Sigmoid"
weight_filler = Filler('msra')
filler_constant = Filler('constant')

weight_decay = 0
if name == 'full':
    conv_type = "Convolution"
    weight_decay = 1e-4
elif name == 'tb':
    conv_type = "TBConvolution"
else:
    conv_type = 'XnorNetConvolution'

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU(1)
max_iter = num_epoch * cifar10.train_iter
solver.test(test_iter=cifar10.test_iter, test_interval=2 * cifar10.train_iter,
            test_initialization=False)
solver.train(base_lr=0.01, lr_policy='step',
             # stepsize=10000,
             stepsize=num_epoch * cifar10.train_iter / 3,
             gamma=0.1, max_iter=num_epoch * cifar10.train_iter,
             weight_decay=weight_decay)
# solver.optimizer(type='SGD', momentum=0.9)
solver.optimizer(type='Adam')
solver.display(display=200, average_loss=200)
solver.snapshot(snapshot=10000, snapshot_prefix='snapshot/' + name)

# --------- Network ----------
# 32C5 - sigmoid - MaxPool3 - LRN - 32C5 - sigmoid - MaxPool3 - LRN
#  - 64C5 - sigmoid - AvePool3 - 10FC - Softmax
Net("cifar10_" + name)
out, label = cifar10.data()
# out = BN(out, name='bn1')
# out = Activation(out, name='binary1', method="Binary")
out = Conv(out, name='conv1', num_output=32, bias_term=False, kernel_size=3,
           stride=1, pad=1, weight_filler=weight_filler)
out = BN(out, name='bn_act1', bias_term=True, inplace=True)
out = Activation(out, name='act1', method="Sigmoid")
out = Pool(out, name='pool1', method=Net.MaxPool, kernel_size=3)
out = LRN(out, name='lrn1')

out = BN(out, name='bn2')
out = Ternary(out, name='ternary2')
out = Conv(out, name='conv2', conv_type=conv_type, num_output=32,
           bias_term=False, kernel_size=3, stride=1, pad=1,
           weight_filler=weight_filler)
out = BN(out, name='bn_act2', bias_term=True, inplace=True)
out = Activation(out, name='act2', method="Sigmoid")
out = Pool(out, name='pool2', method=Net.MaxPool, kernel_size=3)
out = LRN(out, name='lrn2')

out = BN(out, name='bn3')
out = Ternary(out, name='ternary3')
out = Conv(out, name='conv3', conv_type=conv_type, num_output=64,
           bias_term=False, kernel_size=3, stride=1, pad=1,
           weight_filler=weight_filler)
out = BN(out, name='bn_act3', bias_term=True, inplace=True)
out = Activation(out, name='act3', method="Sigmoid")
out = Pool(out, name='pool3', method=Net.AveragePool, kernel_size=3)

# out = BN(out, name='bn_fc')
out = FC(out, name='fc', num_output=10, weight_filler=weight_filler,
         bias_term=True, bias_filler=filler_constant)
accuracy = Accuracy(out + label)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/cnn2/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
