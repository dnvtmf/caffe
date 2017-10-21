from caffe_user import *
import os

# ----- Configuration -----
name = "tb"
num_epoch = 120
batch_size = 100
cifar10 = CIFAR_10(batch_size)
activation_method = "Sigmoid"
weight_filler = Filler('xavier')
filler_constant = Filler('constant')

weight_decay = 0
t = None
if name == 'full':
    conv = NormalBlock
    weight_decay = 1e-4
elif name == 'tb':
    t = 0.5
    conv = TernaryBlock
else:
    conv = BinaryBlock

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU(0)
max_iter = num_epoch * cifar10.train_iter
solver.test(test_iter=cifar10.test_iter, test_interval=2 * cifar10.train_iter,
            test_initialization=False)
solver.train(base_lr=0.1, lr_policy='multistep',
             stepvalue=[60 * cifar10.train_iter, 90 * cifar10.train_iter],
             gamma=0.1, max_iter=num_epoch * cifar10.train_iter,
             weight_decay=weight_decay)
solver.optimizer(type='SGD', momentum=0.9)
# solver.optimizer(type='Adam')
solver.display(display=200, average_loss=200)
solver.snapshot(snapshot=10000, snapshot_prefix='snapshot/' + name)

# --------- Network ----------
# 32C5 - sigmoid - MaxPool3 - LRN - 32C5 - sigmoid - MaxPool3 - LRN
#  - 64C5 - sigmoid - AvePool3 - 10FC - Softmax
Net("cifar10_" + name)
out, label = cifar10.data()
# out = BN(out, name='bn1')
# out = Activation(out, name='binary1', method="Binary")
out = NormalBlock(out, 'conv1', num_output=32, kernel_size=3, stride=1, pad=1,
                  weight_filler=weight_filler, act="Sigmoid")
out = Pool(out, name='pool1', method=Net.MaxPool, kernel_size=3)
out = LRN(out, name='lrn1')

out = conv(out, 'conv2', num_output=32, kernel_size=3, stride=1, pad=1,
           weight_filler=weight_filler, act="Sigmoid", threshold_t=t)
out = Pool(out, name='pool2', method=Net.MaxPool, kernel_size=3)
out = LRN(out, name='lrn2')

out = conv(out, 'conv3', num_output=64, kernel_size=3, stride=1, pad=1,
           weight_filler=weight_filler, act="Sigmoid", threshold_t=t)
out = Pool(out, name='pool3', method=Net.AveragePool, kernel_size=3)

# out = BN(out, name='bn_fc')
out = FC(out, name='fc', num_output=10, weight_filler=weight_filler,
         bias_term=True, bias_filler=filler_constant)
accuracy = Accuracy(out + label)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/cnn2/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
