from caffe_user import *
import os
import time

num_epoch = 500
batch_size = 200
Net("cifar10")
filler_xavier = Filler('xavier')
filler_uniform = Filler('uniform', min=-0.1, max=0.1)
filler_constant = Filler('constant')
data, label = Data([], phase=TRAIN, source="../cifar10_train_lmdb", batch_size=batch_size, backend=Net.LMDB,
                   optional_params=[Transform(mean_file="mean.binaryproto")])
Data([], phase=TEST, source="../cifar10_test_lmdb", batch_size=100, backend=Net.LMDB,
     optional_params=[Transform(mean_file="mean.binaryproto")])
out = [data]
label = [label]
# fc = XnorNetFC
fc = TBFC
# fc = BinFC
# fc = FC
conv = TBConv
# conv = Conv
# 2x(128C3)-MP2-2x(256C3)-MP2-2x(512C3)-MP2-1024FC-SVM
out = Conv(out, name='conv1', num_output=128, bias_term=True, kernel_size=3, stride=1,
           weight_filler=filler_xavier, bias_filler=filler_constant)
out = ReLU(out, name='relu1')
out = BN(out, name='bn1')
out = conv(out, name='conv2', num_output=128, bias_term=True, kernel_size=3, stride=1,
           weight_filler=filler_xavier, bias_filler=filler_constant)
out = ReLU(out, name='relu2')
out = Pool(out, name='pool1')

out = BN(out, name='bn3')
out = conv(out, name='conv3', num_output=256, bias_term=True, kernel_size=3, stride=1,
           weight_filler=filler_xavier, bias_filler=filler_constant, full_train=True)
out = ReLU(out, name='relu3')
out = BN(out, name='bn4')
out = conv(out, name='conv4', num_output=256, bias_term=True, kernel_size=3, stride=1,
           weight_filler=filler_xavier, bias_filler=filler_constant, full_train=True)
out = ReLU(out, name='relu4')
out = Pool(out, name='pool2')

out = BN(out, name='bn5')
out = conv(out, name='conv5', num_output=512, bias_term=True, kernel_size=3, stride=1,
           weight_filler=filler_xavier, bias_filler=filler_constant, full_train=True)
out = ReLU(out, name='relu5')
out = BN(out, name='bn6')
out = conv(out, name='conv6', num_output=512, bias_term=True, kernel_size=3, stride=1,
           weight_filler=filler_xavier, bias_filler=filler_constant, full_train=True)
out = ReLU(out, name='relu6')
out = Pool(out, name='pool3')

out = BN(out, name='bn7')
out = fc(out, name='fc7', num_output=1024, bias_term=True, weight_filler=filler_xavier, bias_filler=filler_constant,
         full_train=True)
out = ReLU(out, name='relu8')
out = FC(out, name='fc8', num_output=10, weight_filler=filler_xavier, bias_term=True, bias_filler=filler_constant)
accuracy = Accuracy(out + label)
loss = HingeLoss(out + label, norm=2)
# loss = SoftmaxWithLoss(out + label)

# ---------- solver ----
solver = Solver().net('./model.prototxt').CPU()
solver.test(test_iter=100, test_interval=1000, test_initialization=False)
num_iter = num_epoch * 50000 / batch_size
solver.train(base_lr=0.001, lr_policy='step', gamma=0.1, stepsize=num_iter / 5, max_iter=num_iter)
# solver.optimizer(type='SGD', momentum=0.9)
solver.optimizer(type='Adam')
solver.display(display=200)
solver.snapshot(snapshot=10000, snapshot_prefix='cifar10')

sh_content = """#!/usr/bin/env sh
set -e

~/caffe/build/tools/caffe train --solver solver.prototxt $@

"""

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/cnn')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
# print solver.get_solver_proto()
# print get_prototxt()

with open(os.path.join(model_dir, 'model.prototxt'), 'w') as f:
    f.write(get_prototxt())

with open(os.path.join(model_dir, 'solver.prototxt'), 'w') as f:
    f.write(solver.get_solver_proto())

with open(os.path.join(model_dir, 'train.sh'), 'w') as f:
    f.write(sh_content)
os.chmod(os.path.join(model_dir, 'train.sh'), 0777)

print 'time: ' + time.ctime()
print 'successful create model.prototxt, solver.prototxt and train.sh in directory: ' + model_dir
