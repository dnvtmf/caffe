from caffe_layer import *
from caffe_solver import *
from caffe_user import *
import os
import time

Net("mnist_MLP")
filler_xavier = Filler('xavier')
filler_constant = Filler('constant')
data, label = Data([], phase=TRAIN, source="../mnist_train_lmdb", batch_size=100, backend=Net.LMDB,
                   optional_params=[Transform(scale=0.00390625)])
Data([], phase=TEST, source="../mnist_test_lmdb", batch_size=100, backend=Net.LMDB,
     optional_params=[Transform(scale=0.00390625)])
data = [data]
label = [label]
out = BN(data, name='bn0')
fc1 = BinFC(out, name='fc1', num_output=2048, weight_filler=filler_xavier)
bn1 = BN(fc1, name='bn1')
relu1 = ReLU(bn1, name='relu1')
fc2 = FC(relu1, name='fc2', num_output=10, weight_filler=filler_xavier)
accuracy = Accuracy(fc2 + label)
loss = HingeLoss(fc2 + label, norm=2)
# loss = SoftmaxWithLoss(ip2 + [label])

# ---------- solver ----
solver = Solver().net('./model.prototxt').CPU()
solver.test(test_iter=100, test_interval=500, test_initialization=False)
solver.train(base_lr=1e-3, lr_policy='step', stepsize=500, gamma=0.1, weight_decay=1e-4, max_iter=1000)
solver.optimizer(type='Adam', momentum=0.9, momentum2=0.999)
solver.display(display=100)
solver.snapshot(snapshot=5000, snapshot_prefix='binary_lenet')

sh_content = """
#!/usr/bin/env sh
set -e

~/caffe/build/tools/caffe train --solver solver.prototxt $@

"""

model_dir = '/home/wan/mnist/mlp_model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
# print solver.get_solver_proto()
print get_prototxt()

with open(os.path.join(model_dir, 'model.prototxt'), 'w') as f:
    f.write(get_prototxt())

with open(os.path.join(model_dir, 'solver.prototxt'), 'w') as f:
    f.write(solver.get_solver_proto())

with open(os.path.join(model_dir, 'train.sh'), 'w') as f:
    f.write(sh_content)
os.chmod(os.path.join(model_dir, 'train.sh'), 0777)

print 'time: ' + time.ctime()
print 'successful create model.prototxt, solver.prototxt and train.sh in directory: ' + model_dir
