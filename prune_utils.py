import torch
import torch.nn as nn


class Pruner:
    def __init__(self, net):
        self.net = net.eval()
        # Initialize stuff
        self.clear_rank()
        self.clear_modules()
        self.clear_cache()
        # Set hooks
        self._register_hooks()

    def clear_rank(self):
        self.ranks = {}  # accumulates Taylor ranks for modules
        self.num_batches = 0  # how many minibatches so far

    def clear_modules(self):
        self.convs = []
        self.BNs = []

    def clear_cache(self):
        self.activation_maps = []
        self.gradients = []

    def _register_hooks(self):
        def forward_hook_fn(module, input, output):
            """ Stores the forward pass outputs (activation maps)"""
            self.activation_maps.append(output.clone().detach())

        def backward_hook_fn(module, grad_in, grad_out):
            """Stores the gradients wrt outputs during backprop"""
            self.gradients.append(grad_out[0].clone().detach())

        for name, module in self.net.named_modules():
            if isinstance(module, nn.Conv2d):
                if name != "outc.conv":  # don't hook final conv module
                    module.register_backward_hook(backward_hook_fn)
                    module.register_forward_hook(forward_hook_fn)
                self.convs.append(module)
            if isinstance(module, nn.BatchNorm2d):
                self.BNs.append(module)  # save corresponding BN layer

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

    def _rank_channels(self, prune_channels):
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
        sorted_rank, sorted_indices = torch.topk(total_rank, prune_channels, largest=False)
        sorted_channel_layers = channel_layers[sorted_indices]
        sorted_layer_channels = layer_channels[sorted_indices]
        return sorted_channel_layers, sorted_layer_channels

    def pruning(self, prune_channels):

        sorted_channel_layers, sorted_layer_channels = self._rank_channels(prune_channels)
        inchans, outchans = self.create_indices()

        for i in range(len(sorted_channel_layers)):
            cl = int(sorted_channel_layers[i])
            lc = int(sorted_layer_channels[i])

            # These tensors are concat at a later conv2d
            # res_prev = {1:16, 3:14, 5:12, 7:10}
            res = True if cl in [1, 3, 5, 7] else False

            # These tensors are concat with an earlier tensor at bottom.
            offset = True if cl in [9, 11, 13, 15] else False

            # TODO: Wrong number of channels at up1 input
            # Remove indices of pruned parameters/channels
            if offset:
                bottom = len(outchans[cl])
                try:
                    inchans[cl + 1].remove(lc - bottom)  # it is searching for a -ve number to remove, but there are none
                    # However, the output channel of the previous layer (d4) is reduced
                    # So up1's input channel is larger than expected due to failed removal
                except ValueError:
                    pass
            else:
                try:
                    inchans[cl + 1].remove(lc)
                except ValueError:
                    pass
            if res:
                try:
                    inchans[-(cl + 2)].remove(lc)
                except ValueError:
                    pass
            try:
                outchans[cl].remove(lc)
            except ValueError:
                pass

        # Use indexing to get rid of parameters
        for i, c in enumerate(self.convs):
            self.convs[i].weight.data = c.weight[outchans[i], ...][:, inchans[i], ...]
            self.convs[i].bias.data = c.bias[outchans[i]]

        for i, bn in enumerate(self.BNs):
            self.BNs[i].running_mean.data = bn.running_mean[outchans[i]]
            self.BNs[i].running_var.data = bn.running_var[outchans[i]]
            self.BNs[i].weight.data = bn.weight[outchans[i]]
            self.BNs[i].bias.data = bn.bias[outchans[i]]

    def create_indices(self):
        chans = [(list(range(c.weight.shape[1])), list(range(c.weight.shape[0]))) for c in self.convs]
        inchans, outchans = list(zip(*chans))
        return inchans, outchans

    def channel_save(self, path):
        """save the 22 distinct number of channels"""
        chans = []
        for i, c in enumerate(self.convs[1:-1]):
            if (i > 8 and (i-9) % 2 == 0) or i == 0:
                chans.append(c.weight.shape[1])
            chans.append(c.weight.shape[0])

        with open(path, 'w') as f:
            for item in chans:
                f.write("%s\n" % item)