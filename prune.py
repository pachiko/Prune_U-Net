import torch
import torch.nn as nn
from torch import optim
import numpy as np

import os
import os.path as osp
import json
from optparse import OptionParser
from prune_utils import Pruner
from tqdm import tqdm
from finetune import finetune
from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, get_logger, get_save_dir


def get_args():
    parser = OptionParser()
    parser.add_option('-n', '--name', dest='name',
                      default="initial", help='run name')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=2,
                      type='int', help='batch size')
    parser.add_option('-t', '--taylor_batches', dest='taylor_batches', default=2,
                      type='int', help='number of mini-batches used to calculate Taylor criterion')
    parser.add_option('-p', '--prune_channels', dest='prune_channels', default=200,
                      type='int', help='number of channels to remove')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-c', '--channel_txt', dest='channel_txt',
                      default="model_channels.txt", help='load channel txt')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-r', '--lr', dest='lr', type='float',
                      default=0.1, help='learning rate for finetuning')
    parser.add_option('-i', '--iters', dest='iters', type='int',
                      default=100, help='number of mini-batches for fine-tuning')
    parser.add_option('-e', '--epochs', dest='epochs', type='int',
                      default=5, help='number of epochs for final finetuning')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':

    # Book-keeping & paths
    args = get_args()

    dir_img = 'data/train/'
    dir_mask = 'data/train_masks/'
    dir_checkpoint = 'save/'
    splitfile = "data/trainval.json"

    runname = args.name
    save_path = os.path.join(dir_checkpoint, runname)
    save_dir = get_save_dir(save_path, runname, training=False)  # unique save dir
    log = get_logger(save_dir, runname)  # logger
    log.info('Args: {}'.format(json.dumps({"batch_size": args.batch_size,
                                           "taylor_batches": args.taylor_batches,
                                           "prune_channels": args.prune_channels,
                                           "gpu": args.gpu,
                                           "load": args.load,
                                           "channel_txt": args.channel_txt,
                                           "scale": args.scale,
                                           "lr": args.lr,
                                           "iters": args.iters,
                                           "epochs": args.epochs},
                                          indent=4, sort_keys=True)))

    # Dataset
    if not os.path.exists(splitfile):  # Our constant datasplit
        ids = get_ids(dir_img)  # [file1, file2]
        ids = split_ids(ids)  # [(file1, 0), (file1, 1), (file2, 0), ...]
        iddataset = split_train_val(ids, 0.2, splitfile)
        log.info("New split dataset")

    else:
        with open(splitfile) as f:
            iddataset = json.load(f)
        log.info("Load split dataset")

    train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, args.scale)
    val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, args.scale)

    # Model Initialization
    net = UNet(n_channels=3, n_classes=1, f_channels=args.channel_txt)
    log.info("Built model using {}...".format(args.channel_txt))
    if args.gpu:
        net.cuda()
    if args.load:
        net.load_state_dict(torch.load(args.load))
        log.info('Loading checkpoint from {}...'.format(args.load))

    pruner = Pruner(net)  # Pruning handler
    criterion = nn.BCELoss()

    # Ranking on the train dataset
    log.info("Evaluating Taylor criterion for %i mini-batches" % args.taylor_batches)
    with tqdm(total=args.taylor_batches*args.batch_size) as progress_bar:
        for i, b in enumerate(batch(train, args.batch_size)):

            net.zero_grad()  # Zero gradients. DO NOT ACCUMULATE

            # Data & Label
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            if args.gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            # Forward pass
            masks_pred = net(imgs).squeeze()

            # Backward pass
            loss = criterion(masks_pred, true_masks)
            loss.backward()

            # Compute Taylor rank
            pruner.compute_rank()

            # Tracking progress
            progress_bar.update(args.batch_size)
            if i % 200 == 0 and i > 200:
                log.info("Evaluated Taylor criterion for %i mini-batches" % i)
            if i == args.taylor_batches:  # Stop evaluating after sufficient mini-batches
                log.info("Finished computing Taylor criterion")
                break

    # Prune & save
    pruner.pruning(args.prune_channels)
    log.info('Completed Pruning of %i channels' % args.prune_channels)

    save_file = osp.join(save_dir, "Pruned.pth")
    torch.save(net.state_dict(), save_file)
    log.info('Saving pruned to {}...'.format(save_file))

    save_txt = osp.join(save_dir, "pruned_channels.txt")
    pruner.channel_save(save_txt)
    log.info('Pruned channels to {}...'.format(save_txt))

    # TODO: Finetuning
    del net, pruner
    net = UNet(n_channels=3, n_classes=1, f_channels=args.channel_txt)  # TODO: change back to save_txt
    log.info("Re-Built model using {}...".format(save_txt))
    if args.gpu:
        net.cuda()
    if args.load:
        net.load_state_dict(torch.load(save_file))
        log.info('Re-Loaded checkpoint from {}...'.format(save_file))

    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    # reset the generators
    train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, args.scale)
    save_file = osp.join(save_dir, "Finetuned.pth")
    finetune(net, optimizer, criterion, train, log, save_file,
             args.iters, args.batch_size, args.gpu)

    # TODO: Add evaluation loop here
    # epoch_loss = eval_net(net, train, len(iddataset['train']), args.gpu, args.batch_size, is_loss=True)
    # log.info('Training Loss: {}'.format(epoch_loss))

    val_dice = eval_net(net, val,  len(iddataset['val']), args.gpu, args.batch_size)
    log.info('Validation Dice Coeff: {}'.format(val_dice))


