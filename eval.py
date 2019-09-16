import torch
import torch.nn as nn

from tqdm import tqdm
import os
import json
from optparse import OptionParser
from prune_utils import Pruner

import numpy as np
from dice_loss import dice_coeff
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, get_logger, get_save_dir
import gc


def eval_net(net, dataset, lendata, gpu=False, batch_size=8, is_loss=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    criterion = nn.BCELoss()
    with torch.no_grad(), tqdm(total=lendata) as progress_bar:
        for i, b in enumerate(batch(dataset, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
                masks_pred = net(imgs).squeeze()

            if is_loss:
                loss = criterion(masks_pred, true_masks)
                tot += loss.item()
                progress_bar.update(args.batch_size)
                progress_bar.set_postfix(BCE=loss.item())
            else:
                masks_pred = (masks_pred > 0.5).float()
                dice = dice_coeff(masks_pred, true_masks).item()
                tot += dice
                progress_bar.update(batch_size)
                progress_bar.set_postfix(DICE=dice)

    value = tot / i
    return value


def get_args():
    parser = OptionParser()
    parser.add_option('-b', '--batch_size', dest='batch_size', default=10,
                      type='int', help='batch size')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    dir_img = 'data/train/'
    dir_mask = 'data/train_masks/'
    # dir_checkpoint = 'checkpoints/'
    splitfile = "data/trainval.json"

    # runname = "initial"
    # save_path = os.path.join("save", runname)
    # save_dir = get_save_dir(save_path, runname, training=False)  # unique save dir
    # log = get_logger(save_dir, runname)  # logger
    # log.info('Args: {}'.format(json.dumps({"batch_size": args.batch_size, "scale": args.scale \
    #         }, indent=4, sort_keys=True)))

    if not os.path.exists(splitfile):  # Our constant datasplit
        pass
        # ids = get_ids(dir_img)  # [file1, file2]
        # ids = split_ids(ids)  # [(file1, 0), (file1, 1), (file2, 0), ...]
        # iddataset = split_train_val(ids, 0.2, splitfile)
        # log.info("New split dataset")
    else:
        with open(splitfile) as f:
            iddataset = json.load(f)
        # log.info("Load split dataset")

    train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, args.scale)
    val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, args.scale)

    net = UNet(n_channels=3, n_classes=1)
    if args.gpu:
        net.cuda()
    net.load_state_dict(torch.load("TEST.pth"))

    # pruner = Pruner(net)
    criterion = nn.BCELoss()

    for i, b in enumerate(batch(train, args.batch_size)):
        net.zero_grad()
        imgs = np.array([i[0] for i in b]).astype(np.float32)
        true_masks = np.array([i[1] for i in b])

        imgs = torch.from_numpy(imgs)
        true_masks = torch.from_numpy(true_masks)

        if args.gpu:
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

        masks_pred = net(imgs).squeeze()

        loss = criterion(masks_pred, true_masks)
        loss.backward()
        # TODO: Seems like network definition doesn't change with parameters pruned
        # pruner.compute_rank()

        # print(i)
        # if i == 4:
        #     pruner.pruning()
        #     torch.save(net.state_dict(), "TEST.pth")
            # for name, module in net.named_modules():
            #     if isinstance(module, nn.Conv2d):
            #         print(module.weight.shape)

    # pruner.pruning()

    # log.info("Built model")

    # if args.gpu:
    #     net.cuda()
    #
    # if args.load:
    #     net.load_state_dict(torch.load(args.load))
    #     log.info('Loading checkpoint from {}...'.format(args.load))

    # epoch_loss = eval_net(net, train, len(iddataset['train']), args.gpu, args.batch_size, is_loss=True)
    # log.info('Training Loss: {}'.format(epoch_loss))

    # val_dice = eval_net(net, val,  len(iddataset['val']), args.gpu, args.batch_size)
    # log.info('Validation Dice Coeff: {}'.format(val_dice))



