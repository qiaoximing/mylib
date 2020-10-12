import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numbers, math

class Trigger():
    def __init__(self, dataset, size=5, type_='solid', target='random', loc='random', args={}):
        super().__init__()
        self.size = (dataset.data_shape[0], size, size)
        self.type_ = type_
        self.loc = loc
        self.args = args
        if target == 'random':
            self.target = torch.randint(dataset.num_classes, (1,)).item()
        if type_ == 'solid':
            self.content = torch.ones(self.size)
        elif type_ == 'bernoulli':
            mean = 0.5
            self.content = torch.bernoulli(torch.ones(self.size) * mean)
        elif type_ == 'gaussian':
            mean, std = 0.5, 0.25
            self.content = F.hardtanh(torch.randn(self.size) * std + mean, 0, 1)
        self.content_normalized = dataset.normalizer(self.content)
    
    def plot(self):
        plt.figure()
        plt.imshow(transforms.ToPILImage()(self.content))
        plt.show()

    def apply_by_index(self, data, label, index):
        batchsize, num_channels, img_h, img_w = data.size()
        _, size_h, size_w = self.size
        indexsize = len(index)
        if self.loc == 'random':
            loc_h = torch.randint(img_h - size_h, (indexsize,))
            loc_w = torch.randint(img_w - size_w, (indexsize,))
        else:
            # self.loc is a 2-int-tuple
            loc_h = torch.Tensor(indexsize).fill_(self.loc(0))
            loc_w = torch.Tensor(indexsize).fill_(self.loc(1))
        idx_h = loc_h.view(-1, 1, 1, 1) + torch.arange(size_h).view(1, 1, -1, 1)
        idx_w = loc_w.view(-1, 1, 1, 1) + torch.arange(size_w).view(1, 1, 1, -1)
        data, label = data.clone(), label.clone()
        data[index.view(-1, 1, 1, 1), torch.arange(num_channels).view(1, -1, 1, 1),
             idx_h, idx_w] = self.content_normalized.view(-1, *self.size)
        label[index] = self.target
        return data, label

    def apply_by_ratio(self, data, label, ratio=1.):
        index = torch.nonzero(torch.rand(data.size(0)) < ratio)
        return self.apply_by_index(data, label, index)

    def apply_by_ratio_fn(self, ratio=1.):
        def fn(data, label):
            return self.apply_by_ratio(data, label, ratio)
        return fn


class Mask():
    def __init__(self, dataset, num_masks, num_covers, batchsize, device, args={}):
        super().__init__()
        self.device = device
        self.size = (batchsize, num_masks, dataset.data_shape[1], dataset.data_shape[2])
        self.num_covers = num_covers
        self.content_raw = torch.randn(*self.size)
        if dataset.data_name == 'imagenet':
            self.args = {
                'r1': 7,
                'r2': 5,
                'downsample': 4
            }
        else:
            self.args = {
                'r1': 5,
                'r2': 3,
                'downsample': 1
            }
        self.args.update({'z': 1})
        self.args.update(args)

    def optimize(self, num_steps=100, lr=10):
        bs, m, n, k = self.size[0], self.size[1], self.size[2] // self.args['downsample'], self.num_covers
        x = torch.randn(bs, m, n, n)
        x.to(self.device)
        x.requires_grad = True
        if k > m / 2:
            k = m - k
        r1, r2, z = self.args['r1'], self.args['r2'], self.args['z']
        kernel_size = 3
        opt = torch.optim.SGD([x], lr=lr)
        for step in range(num_steps):
            opt.zero_grad()
            xp = softtopk(x, k, dim=1)
            g1 = GaussianSmoothing(m, min(kernel_size * r1, n), (r1/z, r1))
            g2 = GaussianSmoothing(m, min(kernel_size * r2, n), (r2, r2/z))
            x1 = g1(xp).view(bs, m, n * n)
            x2 = g2(xp).view(bs, m, n * n)
            l1 = torch.pow(x1, 2).sum(2)
            l2 = torch.bmm(x2, x2.transpose(1, 2)) * (1 - torch.eye(m, m))
            l = l1.sum() + l2.sum()
            l.backward()
            opt.step()
        x = x.view(bs, m, n, 1, n, 1).repeat(1, 1, 1, self.args['downsample'], 1, self.args['downsample']).view(*self.size)
        self.content_raw = x.to('cpu')

    def postprocess(self, trigger_size):
        bs, m, n, k = self.size[0], self.size[1], self.size[2], self.num_covers
        if k > m / 2:
            index = (-self.content_raw).argsort(dim=1)
        else:
            index = self.content_raw.argsort(dim=1)
        self.content = torch.zeros_like(self.content_raw)
        # expansion radius
        r_left = trigger_size // 2
        r_right = r_left
        for b in range(bs):
            for i in range(k):
                for x in range(n):
                    xl = max(0, x - r_left) # left inclusion
                    xr = min(x + r_right + 1, n) # right non-inclusion
                    for y in range(n):
                        yl = max(0, y - r_left)
                        yr = min(y + r_right + 1, n)
                        self.content[b, index[b, m-1-i, x, y], xl:xr, yl:yr] = 1
                for x in range(n):
                    for y in range(n):
                        self.content[b, index[b, m-1-i, x, y], x, y] = 2

    def apply_all(self, data, label):
        bs, m, n = self.size[0], self.size[1], self.size[2]
        mask = 1 - F.hardtanh(self.content, 0, 1).view(bs, m, 1, 1, n, n)
        return data.repeat(bs, m, 1, 1, 1, 1) * mask, label

    def apply_random(self, data, label):
        bs, m, n = self.size[0], self.size[1], self.size[2]
        rand0 = torch.randint(bs, (1,))[0]
        rand1 = torch.randint(m, (1,))[0]
        mask = 1 - F.hardtanh(self.content[rand0, rand1], 0, 1).view(1, 1, n, n)
        return data * mask, label

    def plot_raw(self):
        m = self.size[1]
        fig, axs = plt.subplots(1, m, figsize=(10,10))
        cmaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greys']
        for i in range(m):
            x = self.content_raw.detach().numpy()[0, i]
            axs[i].imshow(x, cmap=cmaps[i])
        fig.show()

    def plot(self):
        m = self.size[1]
        fig, axs = plt.subplots(1, m, figsize=(10,10))
        cmaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greys']
        for i in range(m):
            x = self.content.detach().numpy()[0, i]
            axs[i].imshow(x, cmap=cmaps[i])
        fig.show()


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = F.pad(input, [self.weight.size(2)//2]*4, mode='reflect')
        return self.conv(input, weight=self.weight, groups=self.groups)


def softtopk(x, k, dim):
    """
    Generalized softmax with output range (0, 1) and sum k
    """
    x = x.transpose(dim, len(x.size()) - 1)
    s = x.size()
    x = x.reshape(-1, s[-1])
    s0, s1 = x.size()
    xm = torch.max(x, axis=1, keepdim=True)[0]
    x = torch.exp(x - xm)
    if k == 1:
        t = x / x.sum(axis=1, keepdims=True)
    if k == 2:
        t = torch.bmm(x.view(-1, s1, 1), x.view(-1, 1, s1))
        t *= 1 - torch.eye(s1, s1).view(1, s1, s1)
        t = t.sum(axis=2)
        t = t / t.sum(axis=1, keepdims=True)
    if k > 2:
        raise RuntimeError(
            'k = {} not supported.'.format(k)
        )
    return t.reshape(s).transpose(dim, len(s) - 1)
