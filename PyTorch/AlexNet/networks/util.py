import torch
import torch.nn as nn


class Ternary(torch.autograd.Function):
    """
    Ternary inputs
    """

    def __init__(self, threshold=0, scale=False, clamp=False):
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

        self.save_for_backward(inputs, output, c_sum, c_num)
        return output, c_sum, c_num

    def backward(self, grad_output, grad_c_sum, grad_c_num):
        inputs, output, c_sum, c_num = self.saved_tensors
        grad_input = grad_output.clone()
        if self.scale:
            grad_input += grad_c_sum.mul(output)
        grad_input[inputs.ge(1)] = 0
        grad_input[inputs.le(-1)] = 0
        return grad_input


class Conv2dTB(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, groups=1, threshold=0,
                 scale=False, clamp=False):
        super(Conv2dTB, self).__init__()
        self.output_channels = output_channels
        self.layer_type = 'Conv2d(TB)'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.scale = scale

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        self.ternary = Ternary(threshold, scale, clamp)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            groups=groups)

        if self.scale:
            self.beta_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.bn(x)
        x, c_sum, c_cnt = self.ternary(x)
        x = self.conv(x)
        if self.conv:
            weight = self.beta_conv.weight.data
            self.beta_conv.weight.data = torch.ones(weight.size()).type_as(weight)
            beta = self.beta_conv(c_sum).div(self.beta_conv(c_cnt))
            x.mul_(beta)

    def get_binary_module(self):
        return self.conv


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, groups=1, threshold=0,
                 scale=False, clamp=False, block_type='TB'):
        if block_type == 'TB':
            self.conv_block = Conv2dTB(input_channels, output_channels, kernel_size, stride, padding, groups, threshold,
                scale, clamp)
        elif block_type == 'normal':
            self.conv_block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                    groups=groups), nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True),
                nn.ReLU(inplace=True))
        elif block_type == 'MobileNet':
            self.conv_block = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                    groups=input_channels), nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True),
                nn.ReLU(inplace=True), nn.Conv2d(input_channels, output_channels, groups=groups),
                nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True), nn.ReLU(inplace=True))
        elif block_type == 'TB_DW':
            self.conv_block = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                    groups=input_channels),
                nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True),
                nn.ReLU(inplace=True),
                Conv2dTB(input_channels, output_channels, 1, 1, 0, groups, threshold, scale, clamp),
                nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True),
                nn.ReLU(inplace=True))
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
        for module in self.target_modules:
            self.saved_params.append(module.weight.data.clone())
            self.target_params.append(module.weight)

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
