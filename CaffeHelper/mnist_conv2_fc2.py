from caffe_layer import *
from caffe_solver import *
import os
import time

Net("mnist")
filler_xavier = Filler('xavier')
filler_constant = Filler('constant')
data, label = Data([], phase=TRAIN, source="../mnist_train_lmdb", batch_size=64, backend=Net.LMDB,
                   optional_params=[Transform(scale=0.00390625)])
Data([], phase=TEST, source="../mnist_test_lmdb", batch_size=100, backend=Net.LMDB,
     optional_params=[Transform(scale=0.00390625)])
conv1 = Conv([data], name="conv1", num_output=32, kernel_size=5, stride=1, weight_filler=filler_xavier,
             bias_filler=filler_constant)
pool1 = Pool(conv1, name='pool1', method=Net.MaxPool, kernel_size=2, stride=2)
pool1 = ReLU(pool1, name='relu')
bn1 = BatchNorm(pool1, name='bn1')
scale1 = Scale(bn1, name='scale1', bias_term=True)
conv2 = Conv(scale1, name='conv2', num_output=64, kernel_size=5, stride=1, weight_filler=filler_xavier,
             bias_filler=filler_constant)
pool2 = Pool(conv2, name='pool2', method=Net.MaxPool, kernel_size=2, stride=2)
bn2 = BatchNorm(pool2, name='bn2')
scale2 = Scale(bn2, name='scale2', bias_term=True)
ip1 = FC(scale2, name='ip1', num_output=512, weight_filler=filler_xavier, bias_filler=filler_constant)
relu1 = ReLU(ip1, name='relu1')
ip2 = FC(relu1, name='ip2', num_output=10, weight_filler=filler_xavier, bias_filler=filler_constant)
accuracy = Accuracy(ip2 + [label])
# loss = HingeLoss(ip2 + [label], norm=2)
loss = SoftmaxWithLoss(ip2 + [label])

# ---------- solver ----
solver = Solver().net('./model.prototxt').CPU().test(test_iter=100, test_interval=500)
solver.train(base_lr=1e-3, lr_policy='fixed', weight_decay=1e-4, max_iter=1000)
solver.optimizer(type='Adam', momentum=0.9, momentum2=0.999)
solver.display(display=100)
solver.snapshot(snapshot=5000, snapshot_prefix='binary_lenet')

sh_content = """
#!/usr/bin/env sh
set -e

~/caffe/build/tools/caffe train --solver solver.prototxt $@

"""

model_dir = '/home/wan/mnist/temp_model'
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
