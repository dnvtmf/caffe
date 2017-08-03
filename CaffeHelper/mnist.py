from caffe_user import *
import os
import time

Net("mnist")
filler_xavier = Filler('xavier')
filler_uniform = filler_xavier  # Filler('uniform', min=-0.1, max=0.1)
filler_constant = Filler('constant')
data, label = Data([], phase=TRAIN, source="../mnist_train_lmdb", batch_size=100, backend=Net.LMDB,
                   optional_params=[Transform(scale=0.00390625)])
Data([], phase=TEST, source="../mnist_test_lmdb", batch_size=100, backend=Net.LMDB,
     optional_params=[Transform(scale=0.00390625)])
out = [data]
label = [label]
# fc = XnorNetFC
fc = TBFC
# fc = BinFC
# fc = FC
conv = TBConv
# conv = Conv
# 32-C5 + MP2 + 64-C5 + MP2 + 512 FC + SVM
out = Conv(out, name='conv1', num_output=32, bias_term=True, kernel_size=5, stride=1,
           weight_filler=filler_xavier, bias_filler=filler_constant)
out = ReLU(out, name='relu1')
out = Pool(out, name='pool1')
out = BN(out, name='bn1')
out = conv(out, name='conv2', num_output=64, bias_term=True, kernel_size=5, stride=1,
           weight_filler=filler_xavier, bias_filler=filler_constant, full_train=True)
out = ReLU(out, name='relu2')
out = Pool(out, name='pool2')
out = BN(out, name='bn2')
out = fc(out, name='fc3', num_output=512, bias_term=True, weight_filler=filler_xavier, bias_filler=filler_constant,
         full_train=True)
out = ReLU(out, name='relu3')
out = FC(out, name='fc4', num_output=10, weight_filler=filler_xavier, bias_term=True, bias_filler=filler_constant)
accuracy = Accuracy(out + label)
# loss = HingeLoss(out + label, norm=2)
loss = SoftmaxWithLoss(out + label)

# ---------- solver ----
solver = Solver().net('./model.prototxt').CPU()
solver.test(test_iter=100, test_interval=500, test_initialization=False)
solver.train(base_lr=0.001, lr_policy='fixed', max_iter=3000)
# solver.train(base_lr=0.001, lr_policy='step', gamma=0.1, stepsize=1000, max_iter=3000)
solver.optimizer(type='SGD', momentum=0.9)
# solver.optimizer(type='Adam')
solver.display(display=100)
solver.snapshot(snapshot=5000, snapshot_prefix='tb')

sh_content = """#!/usr/bin/env sh
set -e

~/caffe/build/tools/caffe train --solver solver.prototxt $@

"""

model_dir = os.path.join(os.getenv('HOME'), 'mnist/mnist_model')
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
