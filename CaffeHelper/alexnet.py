from caffe_user import *
import os

# ----- Configuration -----
name = "Binary"
batch_size = 512
images = ImageNet(batch_size)
weight_filler = Filler('msra')

weight_decay = 0
t = None
conv = TBBlock
tb_method = name
act_method = None
scale_term = True
g = 1
if name == 'full':
    tb_method = "ReLU"
    act_method = None
    conv = NormalBlock
    weight_decay = 5e-4

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU(1)
solver.test(test_iter=images.test_iter, test_interval=1000,
            test_initialization=False)
step_size = 4 * images.train_iter
max_iter = 16 * images.train_iter
solver.train(base_lr=0.01, lr_policy='step', gamma=0.01, stepsize=step_size,
             max_iter=max_iter, weight_decay=weight_decay)
solver.optimizer(type='Adam')
solver.display(display=1, average_loss=1)
solver.snapshot(snapshot=10000, snapshot_prefix='snapshot/' + name)

# --------- Network ----------
Net("ImageNet_" + name)
out, label = images.data(cs=224)

out = NormalBlock(out, 'conv1', None, 96, 11, 4, 2, weight_filler, "ReLU")
out = Pool(out, name='pool1', method=Net.MaxPool, kernel_size=3, stride=2)

out = conv(out, 'conv2', tb_method, 256, 5, 1, 2, weight_filler, act_method,
           threshold_t=t, scale_term=scale_term, group=g)
out = Pool(out, name='pool2', method=Net.MaxPool, kernel_size=3, stride=2)

out = conv(out, 'conv3', tb_method, 384, 3, 1, 1, weight_filler, act_method,
           threshold_t=t, scale_term=scale_term)
out = conv(out, 'conv4', tb_method, 384, 3, 1, 1, weight_filler, act_method,
           threshold_t=t, scale_term=scale_term, group=g)
out = conv(out, 'conv5', tb_method, 256, 3, 1, 1, weight_filler, act_method,
           threshold_t=t, scale_term=scale_term, group=g)
out = Pool(out, name='pool5', method=Net.MaxPool, kernel_size=3, stride=2)

out = conv(out, 'fc6', tb_method, 4096, 6, 1, 0, weight_filler, act_method,
           threshold_t=t, scale_term=scale_term)
out = conv(out, 'fc7', tb_method, 4096, 1, 1, 0, weight_filler, "ReLU",
           threshold_t=t, scale_term=scale_term)

out = FC(out, name='fc8', num_output=1000, weight_filler=weight_filler)

accuracy = Accuracy(out + label)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'alexnet/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
