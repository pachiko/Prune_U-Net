import torch.nn as nn


def flops_count(net, size):
    H, W = size
    flops = []
    is_warp = True

    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) and name != "outc.conv":
            O, I, KH, KW = module.weight.shape

            if is_warp:
                if name[:2] == "up":
                    H *= 2
                    W *= 2
                elif name[:4] == "down":
                    H //= 2
                    W //= 2

            flops.append(H * W * KH * KW * I)
            is_warp = not is_warp

    return flops
