from caffe_user import *
import os

# ----- Configuration -----
name = "Ternary"
batch_size = 128
images = ImageNet(batch_size)
weight_filler = Filler('msra')
bias_filler = Filler('constant')

weight_decay = 0
t = 0.8
conv = TBBlock
tb_method = name
act_method = "TanH"
scale_term = True
if name == 'full':
    tb_method = "ReLU"
    act_method = None
    conv = NormalBlock
    weight_decay = 5e-4

# ---------- solver ----
solver = Solver().net('./model.prototxt').GPU(1)
solver.test(test_iter=images.num_test / batch_size, test_interval=1000,
            test_initialization=False)
solver.train(base_lr=0.001, lr_policy='step', gamma=0.1,
             stepsize=100000, max_iter=450000,
             weight_decay=weight_decay)
solver.optimizer(type='Adam')
solver.display(display=20, average_loss=20)
solver.snapshot(snapshot=10000, snapshot_prefix='snapshot/' + name)

# --------- Network ----------
Net("ImageNet_" + name)
out, label = images.data(cs=227)

out = NormalBlock(out, 'conv1', None, 96, 11, 4, 0, weight_filler, act_method)
out = Pool(out, name='pool1', method=Net.MaxPool, kernel_size=3, stride=2)

out = conv(out, 'conv2', tb_method, 256, 5, 1, 2, weight_filler, act_method,
           threshold_t=t, scale_term=scale_term, group=2)
out = Pool(out, name='pool2', method=Net.MaxPool, kernel_size=3, stride=2)

out = conv(out, 'conv3', tb_method, 384, 3, 1, 1, weight_filler, act_method,
           threshold_t=t, scale_term=scale_term)
out = conv(out, 'conv4', tb_method, 384, 3, 1, 1, weight_filler, act_method,
           threshold_t=t, scale_term=scale_term, group=2)
out = conv(out, 'conv5', tb_method, 256, 3, 1, 1, weight_filler, act_method,
           threshold_t=t, scale_term=scale_term, group=2)
out = Pool(out, name='pool5', method=Net.MaxPool, kernel_size=3, stride=2)

out = conv(out, 'fc6', tb_method, 4096, 6, 1, 0, weight_filler, act_method,
           threshold_t=t, scale_term=scale_term)
out = conv(out, 'fc7', tb_method, 4096, 1, 1, 0, weight_filler, "ReLU",
           threshold_t=t, scale_term=scale_term)

out = FC(out, name='fc8', num_output=1000, weight_filler=weight_filler,
         bias_filler=bias_filler)

accuracy = Accuracy(out + label)
# loss = HingeLoss(out + label, norm=2)
loss = SoftmaxWithLoss(out + label)

model_dir = os.path.join(os.getenv('HOME'), 'alexnet/' + name)
gen_model(model_dir, solver, [0, 2, 4, 6])
