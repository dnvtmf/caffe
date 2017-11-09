import os
import torch
import torch.nn as nn
import torch.nn.init as init


class Ternary(torch.autograd.Function):
    """
    Ternary inputs
    """

    def __init__(self, threshold=0., scale=False, clamp=False, **kwargs):
        super(Ternary, self).__init__(**kwargs)
        self.threshold = threshold
        self.scale = scale
        self.clamp = clamp

    def forward(self, inputs):
        size = inputs.size()
        output = torch.zeros(size).type_as(inputs)
        output[inputs.ge(self.threshold)] = 1
        output[inputs.le(-self.threshold)] = -1
        c_sum = torch.Tensor()
        c_num = torch.Tensor()
        if self.scale:
            if self.clamp:
                inputs.clamp_(-1, 1)
            c_sum = inputs.mul(output).sum(1, keepdim=True)
            c_num = output.norm(1, 1, keepdim=True)
        self.save_for_backward(inputs, output)
        return output, c_sum, c_num

    def backward(self, grad_output, grad_c_sum, grad_c_num):
        inputs, output = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[inputs.ge(1)] = 0
        grad_input[inputs.le(-1)] = 0
        if self.scale:
            grad_input += grad_c_sum.mul(output)
        return grad_input


class Conv2dTB(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, groups=1, threshold=0.6,
            scale=False, clamp=False, bias=True, **kwargs):
        super(Conv2dTB, self).__init__()
        self.output_channels = output_channels
        self.layer_type = 'Conv2d(TB)'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.threshold = threshold
        self.scale = scale
        self.clamp = clamp

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            groups=groups, bias=bias)

        if self.scale:
            self.beta_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.bn(x)
        x, c_sum, c_cnt = Ternary(self.threshold, self.scale, self.clamp)(x)
        x = self.conv(x)
        if self.scale:
            weight = self.beta_conv.weight.data
            self.beta_conv.weight.data = torch.ones(weight.size()).type_as(weight)
            cnt = self.beta_conv(c_cnt)
            cnt[cnt.eq(0)] = 1  # avoid divide zero
            beta = self.beta_conv(c_sum).div(cnt)
            x = x.mul(beta)
        return x

    def get_binary_module(self):
        return self.conv


class RandomTernary(torch.autograd.Function):
    """
    Ternary inputs
    """

    def __init__(self, **kwargs):
        super(RandomTernary, self).__init__(**kwargs)

    def forward(self, inputs):
        output = inputs.add(torch.rand(inputs.size()).type_as(inputs)).floor()
        return output

    def backward(self, grad_output):
        return grad_output


class Conv2dRTB(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, groups=1, is_tanh=True,
            **kwargs):
        super(Conv2dRTB, self).__init__()
        self.output_channels = output_channels
        self.layer_type = 'Conv2d(RTB)'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4)
        if is_tanh:
            self.non_linear = nn.Tanh()
        else:
            self.non_linear = nn.Hardtanh()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            groups=groups)

    def forward(self, x):
        x = self.bn(x)
        x = self.non_linear(x)
        x = RandomTernary()(x)
        x = self.conv(x)
        return x

    def get_binary_module(self):
        return self.conv


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, groups=1, threshold=0,
            scale=False, clamp=False, block_type='TB'):
        super(ConvBlock, self).__init__()
        if block_type == 'TB':
            self.conv_block = nn.Sequential(
                Conv2dTB(input_channels, output_channels, kernel_size, stride, padding, groups, threshold, scale,
                    clamp), nn.BatchNorm2d(output_channels, eps=1e-4), nn.ReLU())
        elif block_type == 'RTB':
            self.conv_block = Conv2dRTB(input_channels, output_channels, kernel_size, stride, padding, groups)
        elif block_type == 'normal':
            self.conv_block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                    groups=groups), nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True),
                nn.ReLU(inplace=True))
        elif block_type == 'MobileNet':
            self.conv_block = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                    groups=input_channels), nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True),
                nn.ReLU(inplace=True), nn.Conv2d(input_channels, output_channels, kernel_size=1, groups=groups),
                nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True), nn.ReLU(inplace=True))
        elif block_type == 'TB_DW':
            self.conv_block = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                    groups=input_channels), nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True),
                nn.ReLU(inplace=True),
                Conv2dTB(input_channels, output_channels, 1, 1, 0, groups, threshold, scale, clamp),
                nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True), nn.ReLU(inplace=True))
        else:
            raise Exception('UNKNOWN Conv Block: %s (TB, normal, MobileNet, TB_DW)' % block_type)

    def forward(self, x):
        x = self.conv_block(x)
        return x


class BinOp:
    def __init__(self, model):
        # get the binary weight of Conv2d and Linear
        self.saved_params = []
        self.target_params = []
        self.target_modules = []

        def get_binary_weight(m):
            if type(m) == Conv2dTB:
                self.target_modules.append(m.get_binary_module())

        model.apply(get_binary_weight)
        self.num_of_params = len(self.target_modules)
        for m in self.target_modules:
            self.saved_params.append(m.weight.data.clone())
            self.target_params.append(m.weight)

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_params[index].data.size()
            negMean = self.target_params[index].data.mean(1, keepdim=True).mul(-1).expand_as(
                self.target_params[index].data)
            self.target_params[index].data = self.target_params[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_params[index].data.clamp(-1.0, 1.0, out=self.target_params[index].data)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_params[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_params[index].data[0].nelement()
            s = self.target_params[index].data.size()
            if len(s) == 4:
                m = self.target_params[index].data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1,
                    keepdim=True).div(n)
            elif len(s) == 2:
                m = self.target_params[index].data.norm(1, 1, keepdim=True).div(n)
            self.target_params[index].data.sign().mul(m.expand(s), out=self.target_params[index].data)

    def restore(self):
        for index in range(self.num_of_params):
            self.target_params[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_params[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0
            m = m.mul(self.target_params[index].grad.data)
            m_add = weight.sign().mul(self.target_params[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_params[index].grad.data = m.add(m_add).mul(1.0 - 1.0 / s[1]).mul(n)
            self.target_params[index].grad.data = self.target_params[index].grad.data.mul(1e+9)


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)


def data_loader(args, input_size=224, caffe_data=False):
    if caffe_data:
        import ImageNet.datasets as datasets
        import ImageNet.datasets.transforms as transforms
        if not os.path.exists(args.data + '/ilsvrc12_mean.binaryproto'):
            print("==> Data directory" + args.data + "does not exits")
            print("==> Please specify the correct data path by")
            print("==>     --data <DATA_PATH>")
            return

        normalize = transforms.Normalize(meanfile=args.data + '/ilsvrc12_mean.binaryproto')

        train_dataset = datasets.ImageFolder(args.data, transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
                transforms.RandomSizedCrop(input_size), ]), Train=True)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data,
            transforms.Compose(
                [transforms.ToTensor(), normalize,
                    transforms.CenterCrop(input_size), ]),
            Train=False),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=True)
    else:
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
        train_dir = os.path.join(args.data, 'train')
        val_dir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(train_dir, transforms.Compose(
            [transforms.RandomSizedCrop(input_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                normalize, ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None), num_workers=args.workers,
            pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(val_dir, transforms.Compose(
            [transforms.Scale(256), transforms.CenterCrop(input_size), transforms.ToTensor(), normalize, ])),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=True)

    return train_loader, val_loader, train_sampler
