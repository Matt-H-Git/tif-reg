import logging
import torch
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

    # Dataset Parameters
    parser.add_argument('--batch-size', default=4, type=int,
                        metavar='N', help='Batch size of the test data')

    # Model saving / loading
    parser.add_argument('-o', '--model-path', default='./models/best.pth', type=str,
                        metavar='BASENAME', help='Model weights to load')

    args = parser.parse_args(argv)
    args.add_noise = True if args.add_noise is not None else False
    return args


# Run through the testing images of ModelNet and accumulate an average loss
def eval(args, evalloader):
    model = TIFReg(args)
    checkpoint = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()
    trainer = TIFTrain(args)

    for param in model.parameters():  # Disable gradients to preserve memory
        param.requires_grad = False

    avg_loss = trainer.eval(model, evalloader)
    rmse_r, mae_r, rmse_t, mae_t = trainer.metrics

    log_message = "Evaluation loss on {}: {:0.4f}\nMetrics: RMSE(R) {:0.4f} | MAE(R) {:0.4f} | RMSE(t) {:0.4f} | MAE(t) {:0.4f}".format(
        args.dataset, avg_loss, rmse_r, mae_r, rmse_t, mae_t)
    LOGGER.info(log_message)
    print(log_message)

def get_dataloader(args):
    evalset = None
    if args.dataset == "modelnet":
        evalset = ModelNet40(args, 'test')
    elif args.dataset == "tum3d":
        evalset = TUM3D(args, 'test')
    evalloader = DataLoader(evalset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return evalloader


if __name__ == "__main__":
    ARGS = parameters()

    if not torch.cuda.is_available():
        ARGS.device = 'cpu'
    ARGS.device = torch.device(ARGS.device)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename="logs/testing.log")

    dataloader = get_dataloader(ARGS)

    eval(ARGS, dataloader)