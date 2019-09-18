import torch
import numpy as np

from tqdm import tqdm
from utils import batch, AverageMeter


def finetune(net, optimizer, criterion, train, log, path, iters=100, batch_size=2, gpu=True):
    net.train()
    bce_meter = AverageMeter()

    with tqdm(total=iters*batch_size) as progress_bar:
        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs).squeeze()

            loss = criterion(masks_pred, true_masks)
            bce_meter.update(loss.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(batch_size)
            progress_bar.set_postfix(BCE=bce_meter.avg)

    log.info("Finetuned loss: {}".format(bce_meter.avg))
    torch.save(net.state_dict(), path)
    log.info('Saving finetuned to {}...'.format(path))