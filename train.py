"""
https://pmc.ncbi.nlm.nih.gov/articles/PMC8434253/pdf/sensors-21-05778.pdf
"""
import logging
import argparse
import os

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from args import global_parameters
from datasets.modelnet40 import ModelNet40
from datasets.tum3d import TUM3D
from tif.model import TIFReg
from trainer import TIFTrain

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def parameters(argv=None):
    parser = global_parameters()

    # TODO Add exp name

    # Training Parameters
    parser.add_argument('--batch-size', default=4, type=int,
                        metavar='N', help='Batch size of the training data')

    # Model saving / loading
    parser.add_argument('--resume', default='', type=str,
                        metavar='BASENAME', help='Full path to the checkpoint to load')
    parser.add_argument('-o', '--outfile', default='./result/tif', type=str,
                        metavar='BASENAME', help='output filename (prefix)')
    args = parser.parse_args(argv)
    args.add_noise = True if args.add_noise is not None else False
    return args

def get_dataloaders(args):
    trainset = None
    evalset = None
    if args.dataset == "modelnet":
        trainset = ModelNet40(args)
        evalset = ModelNet40(args, 'test')
    elif args.dataset == "tum3d":
        trainset = TUM3D(args)
        evalset = TUM3D(args, 'test')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    evalloader = DataLoader(evalset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return trainloader, evalloader


def train(args):
    args.start_epoch = 0
    model = TIFReg(args)
    trainer = TIFTrain(args)
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(learnable_params, lr=0.0001)

    checkpoint = None
    if args.resume and os.path.isfile(args.resume):
        assert args.resume != "{}.pth".format(args.outfile) # Don't want to overwrite the old checkpoint
        checkpoint = torch.load(args.resume, weights_only=False)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        message = "Resuming from pretrained checkpoint: {} at epoch {}".format(args.resume, args.start_epoch)
        print(message)
        LOGGER.info(message)
    model.to(args.device)

    # Dataloaders
    trainloader, evalloader = get_dataloaders(args)

    # Optimiser
    if args.resume and os.path.isfile(args.resume):
        optim.load_state_dict(checkpoint['optimizer'])
    scheduler = MultiStepLR(optim, milestones=[40, 60], gamma=0.1, last_epoch=args.start_epoch-1)
    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(checkpoint['scheduler'])

    min_loss = float('inf')
    epochs = 80 # Determined from the paper
    LOGGER.debug('train, begin')
    is_resume = True if args.resume else False
    LOGGER.info('args: Dropout %f, K neighbours %f, Number of points %f, Max rotation %f, Max translation %f, Resume from checkpoint %s',
                args.dropout, args.k, args.num_points, args.max_rot, args.max_trans, is_resume)
    for epoch in range(args.start_epoch, epochs):
        running_loss = trainer.train(model, trainloader, optim)
        val_loss = trainer.eval(model, evalloader)

        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)

        LOGGER.info('epoch, %04d, %f, %f', epoch + 1, running_loss, val_loss)
        print('epoch, %04d, floss_train=%f, floss_val=%f' % (epoch + 1, running_loss, val_loss))

        if is_best:
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            save_checkpoint(checkpoint, args.outfile)
            print("Saved checkpoint")
            LOGGER.info('Saved checkpoint at epoch %04d' % (epoch + 1))

        scheduler.step()

    LOGGER.debug('train, end')
    print("Finished training!")

def save_checkpoint(state, filename):
    torch.save(state, '{}.pth'.format(filename))

if __name__ == "__main__":
    ARGS = parameters()

    if not torch.cuda.is_available():
        ARGS.device = 'cpu'
    ARGS.device = torch.device(ARGS.device)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename="logs/training.log")

    train(ARGS)