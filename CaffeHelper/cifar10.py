from caffe_user import *
import os

# ----- Configuration -----
name = "Ternary"
num_epoch = 500
batch_size = 50
cifar10 = CIFAR_10(batch_size, True)
weight_filler = Filler('msra')

weight_decay = 0
t = 0.6
conv = TBBlock
tb_method = name
scale_term = True
if name == 'full':
    conv = NormalBlock
    weight_decay = 1e-4

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU(1)
solver.test(test_iter=cifar10.test_iter, test_interval=cifar10.train_iter,
            test_initialization=False)
num_iter = num_epoch * cifar10.train_iter
solver.train(base_lr=0.1, lr_policy='step', gamma=0.5,
             stepsize=50 * cifar10.train_iter,
             max_iter=num_iter, weight_decay=weight_decay)
solver.optimizer(type='SGD', momentum=0.9)
solver.display(display=200, average_loss=200)
solver.snapshot(snapshot=10000, snapshot_prefix='snapshot/' + name)

# --------- Network ----------
# 2x(128C3)-MP2-2x(256C3)-MP2-2x(512C3)-MP2-2x(1024FC)-SVM
Net("cifar10_" + name)
out, label = cifar10.data()

# 3 x 32 x 32
out = Conv(out, name='conv1', num_output=128, bias_term=True, kernel_size=3,
           stride=1, pad=1, weight_filler=weight_filler)
out = TBBlock(out, 'conv2', method=tb_method, scale_term=scale_term,
              num_output=128, kernel_size=3, stride=1, pad=1,
              weight_filler=weight_filler)
out = Pool(out, 'pool2', method=Net.MaxPool)

# 128 x 16 x 16
out = TBBlock(out, 'conv3', method=tb_method, scale_term=scale_term,
              num_output=256, kernel_size=3, stride=1, pad=1,
              weight_filler=weight_filler)
out = TBBlock(out, 'conv4', method=tb_method, scale_term=scale_term,
              num_output=256, kernel_size=3, stride=1, pad=1,
              weight_filler=weight_filler)
out = Pool(out, 'pool4', method=Net.MaxPool)

# 256 x 8 x 8
out = TBBlock(out, 'conv5', method=tb_method, scale_term=scale_term,
              num_output=512, kernel_size=3, stride=1, pad=1,
              weight_filler=weight_filler)
out = TBBlock(out, 'conv6', method=tb_method, scale_term=scale_term,
              num_output=512, kernel_size=3, stride=1, pad=1,
              weight_filler=weight_filler)
out = Pool(out, 'pool6', method=Net.MaxPool)

# 512 x 4 x 4
out = TBBlock(out, 'fc7', method=tb_method, scale_term=scale_term,
              num_output=1024, kernel_size=4, stride=1, pad=0,
              weight_filler=weight_filler)
out = TBBlock(out, 'fc8', method=tb_method, scale_term=scale_term,
              num_output=1024, kernel_size=1, stride=1, pad=0,
              weight_filler=weight_filler)

out = FC(out, 'fc9', num_output=10, weight_filler=weight_filler)
test_accuracy = Accuracy(out + label)
train_accuracy = Accuracy(out + label, test=False)
train_loss = SoftmaxWithLoss(out + label)
# loss = HingeLoss(out + label, norm=2)

model_dir = os.path.join(os.getenv('HOME'), 'cifar10/cnn/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
