import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Pruner:
    def __init__(self, net):
        self.net = net.eval()
        # Initialize stuff
        self.clear_rank()
        self.clear_modules()
        self.clear_cache()
        # Set hooks
        self.register_hooks()

    def clear_rank(self):
        self.ranks = {}  # accumulates Taylor ranks for modules
        self.num_batches = 0  # how many minibatches so far

    def clear_modules(self):
        self.convs = []
        self.convname = []
        self.BNs = []
        self.bnname = []

    def clear_cache(self):
        self.activation_maps = []
        self.gradients = []

    def register_hooks(self):
        def forward_hook_fn(module, input, output):
            """ Stores the forward pass outputs (activation maps)"""
            self.activation_maps.append(output.new_tensor(output))

        def backward_hook_fn(module, grad_in, grad_out):
            """Stores the gradients wrt outputs during backprop"""
            self.gradients.append(grad_out[0].clone().detach())

        for name, module in self.net.named_modules():
            if isinstance(module, nn.Conv2d):
                if name != "outc.conv":  # don't hook final conv module
                    module.register_backward_hook(backward_hook_fn)
                    module.register_forward_hook(forward_hook_fn)
                self.convs.append(module)
                self.convname.append(name)
            if isinstance(module, nn.BatchNorm2d):
                self.BNs.append(module)  # save corresponding BN layer
                self.bnname.append(name)

    def compute_rank(self):  # Compute ranks after each minibatch
        self.num_batches += 1
        self.gradients.reverse()

        for layer, act in enumerate(self.activation_maps):
            taylor = (act*self.gradients[layer]).mean(dim=(2, 3)).abs().mean(dim=0)  # C

            if layer not in self.ranks.keys():  # no such entry
                self.ranks.update({layer: taylor})
            else:
                self.ranks[layer] += taylor  # C
        self.clear_cache()

    def rank_channels(self):
        total_rank = []  # flattened ranks of each channel, all layers
        channel_layers = []  # layer num for each channel
        layer_channels = []  # channel num wrt layer for each channel

        for layer, ranks in self.ranks.items():
            # Average across minibatches
            taylor = ranks/self.num_batches  # C
            # Layer-wise L2 normalization
            taylor = taylor / torch.sqrt(torch.sum(taylor**2))
            total_rank.append(taylor)  # C
            channel_layers.extend([layer]*ranks.shape[0])
            layer_channels.extend(list(range(ranks.shape[0])))

        channel_layers = torch.Tensor(channel_layers)
        layer_channels = torch.Tensor(layer_channels)
        total_rank = torch.cat(total_rank, dim=0)

        # Rank
        sorted_rank, sorted_indices = torch.topk(total_rank, 10, largest=False)
        sorted_channel_layers = channel_layers[sorted_indices]
        sorted_layer_channels = layer_channels[sorted_indices]
        return sorted_channel_layers, sorted_layer_channels

    def pruning(self):
        sorted_channel_layers, sorted_layer_channels = self.rank_channels()
        for i in range(len(sorted_channel_layers)):
            cl = int(sorted_channel_layers[i])
            lc = int(sorted_layer_channels[i])

            prev = self.convs[cl]
            next = self.convs[cl+1]
            bn = self.BNs[cl]
            res = None
            offset = False

            if cl in [1, 3, 5, 7]:  # These tensors are concat at a later conv2d
                # res_prev = {1:16, 3:14, 5:12, 7:10}
                res = self.convs[-(cl+2)]

            if cl in [9, 11, 13, 15]:  # These tensors are concat with an earlier tensor at the bottom.
                offset = True

            new_bn_params = []
            for param in [bn.weight, bn.bias, bn.running_mean, bn.running_var]:
                # param.data = self.remove(param, lc)
                new_bn_params.append(self.remove(param, lc))

            new_bn = nn.BatchNorm2d(new_bn_params[0].shape[0])
            for i, param in enumerate([new_bn.weight, new_bn.bias, new_bn.running_mean, new_bn.running_var]):
                param.data = new_bn_params[i]
            # print(new_bn)
            # self.net._modules[self.bnname[cl]] = new_bn
            # print(self.net)

            # self.BNs[cl] = new_bn
            # self.net.add_module(self.bnname[cl], new_bn)

            # for param in [prev.weight, prev.bias]:
            #     param.data = self.remove(param, lc)
            #
            # if res:  # have residual
            #     res.weight.data = self.remove(res.weight, lc, dim=1)
            #
            # next.weight.data = self.remove(next.weight, -(lc+1) if offset else lc, dim=1)

            break

            # [print(param.shape) for param in [bn.weight, bn.bias, bn.running_mean, bn.running_var]]
            # [print(param.shape) for param in [prev.weight, bn.bias]]
            # [print(param.shape) for param in [next.weight, next.bias]]
            # if res:
            #     [print(param.shape) for param in [res.weight, res.bias]]

    def remove(self, param, lc, dim=0):
        if dim == 0:  # BN params & biases (any vector) OR prev conv (remove filter)
            tmp1 = param[:lc]
            tmp2 = param[lc+1:]

        elif dim == 1:  # next (remove channel)
            tmp1 = param[:, :lc, ...]  # out, in, H, W
            tmp2 = param[:, lc+1:, ...]  # out, in, H, W

        return torch.cat([tmp1, tmp2], dim=dim)
