import torch
import numpy as np

from tqdm import tqdm
from utils import batch, AverageMeter, get_imgs_and_masks
from flops_counter import flops_count


def finetune(net, optimizer, criterion, trainset, log, path, iters=100, epochs=None, batch_size=2, gpu=True, scale=0.5):
    net.train()
    bce_meter = AverageMeter()

    dir_img = 'data/train/'
    dir_mask = 'data/train_masks/'

    if epochs is None:  # Fine-tune using iterations of mini-batches
        epochs = 1
    else:  # Fine-tune using entire epochs
        iters = None

    for e in range(epochs):
        # reset the generators
        train = get_imgs_and_masks(trainset, dir_img, dir_mask, scale)

        with tqdm(total=len(trainset)) as progress_bar:
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
                progress_bar.set_postfix(epoch=e, BCE=bce_meter.avg)

                if i == 0 and e == 0:
                    log.info("FLOPs after pruning: \n{}".format(flops_count(net, imgs.shape[2:])))

                if i == iters:  # Stop finetuning after sufficient mini-batches
                    break

    log.info("Finished finetuning")
    log.info("Finetuned loss: {}".format(bce_meter.avg))
    torch.save(net.state_dict(), path)
    log.info('Saving finetuned to {}...'.format(path))