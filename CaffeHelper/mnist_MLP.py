from caffe_user import *
import os
import time

Net("mnist_MLP")
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
# fc = TBFC
fc = BinFC
# fc = FC
full_train = True
out = BN(out, name='bn0')
out = FC(out, name='fc1', num_output=128, weight_filler=filler_xavier, bias_term=True, bias_filler=filler_constant)
out = BN(out, name='bn1')
out = ReLU(out, name='relu1')
out = BN(out, name='bn_relu1')
out = fc(out, name='fc2', num_output=256, weight_filler=filler_uniform, bias_term=True, bias_filler=filler_constant,
         full_train=full_train)
out = BN(out, name='bn2')
out = ReLU(out, name='relu2')
# out = fc(out, name='fc3', num_output=128, weight_filler=filler_uniform, bias_term=True, bias_filler=filler_constant,
# full_train=full_train)
# out = BN(out, name='bn3')
# out = ReLU(out, name='relu3')
out = FC(out, name='fc4', num_output=10, weight_filler=filler_xavier, bias_term=True, bias_filler=filler_constant)
accuracy = Accuracy(out + label)
# loss = HingeLoss(out + label, norm=2)
loss = SoftmaxWithLoss(out + label)

# ---------- solver ----
solver = Solver().net('./model.prototxt').CPU()
solver.test(test_iter=100, test_interval=500, test_initialization=False)
solver.train(base_lr=0.01, lr_policy='fixed', max_iter=3000)
# solver.train(base_lr=0.001, lr_policy='step', gamma=0.1, stepsize=1000, max_iter=3000, weight_decay=1e-6)
solver.optimizer(type='SGD', momentum=0.9)
# solver.optimizer(type='Adam')
solver.display(display=100)
solver.snapshot(snapshot=5000, snapshot_prefix='binary')

sh_content = """#!/usr/bin/env sh
set -e

~/caffe/build/tools/caffe train --solver solver.prototxt $@

"""

model_dir = os.path.join(os.getenv('HOME'), 'mnist/mlp_model')
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
