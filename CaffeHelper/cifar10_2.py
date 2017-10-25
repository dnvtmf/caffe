from caffe_user import *
import os

# ----- Configuration -----
name = "Ternary"
num_epoch = 120
batch_size = 100
cifar10 = CIFAR_10(batch_size)
act_method = "Sigmoid"
tb_method = name
weight_filler = Filler('msra')

weight_decay = 0
t = 0.6
conv = TBBlock
scale_term = False
if name == 'full':
    conv = NormalBlock
    weight_decay = 1e-4
    tb_method = None

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU(0)
max_iter = num_epoch * cifar10.train_iter
solver.test(test_iter=cifar10.test_iter, test_interval=2 * cifar10.train_iter,
            test_initialization=False)
solver.train(base_lr=0.1, lr_policy='multistep',
             stepvalue=[60 * cifar10.train_iter,  # 1e-2
                        90 * cifar10.train_iter],  # 1e-3
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
out = NormalBlock(out, 'conv1', None, 32, 3, 1, 1, weight_filler, act_method)
out = Pool(out, name='pool1', method=Net.MaxPool, kernel_size=3)  # 16 x 16
out = LRN(out, 'lrn1')

out = conv(out, 'conv2', tb_method, 32, 3, 1, 1, weight_filler, act_method,
           threshold_t=t, scale_term=scale_term)
out = Pool(out, name='pool2', method=Net.MaxPool, kernel_size=3)  # 8 x 8
out = LRN(out, 'lrn2')

out = conv(out, 'conv3', tb_method, 64, 3, 1, 1, weight_filler, act_method,
           threshold_t=t, scale_term=scale_term)
out = Pool(out, name='pool3', method=Net.AveragePool, kernel_size=3)  # 4 x 4

out = FC(out, 'fc', num_output=10, weight_filler=weight_filler)

test_accuracy = Accuracy(out + label)
train_loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/cnn2/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
