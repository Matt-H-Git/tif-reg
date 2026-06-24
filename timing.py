import argparse
import logging
import warnings

from args import global_parameters
from utils import *
import torch.utils.data
import timeit
from datasets.modelnet40 import ModelNet40
from tif.model import TIFReg, TIFRegSimplified
from tqdm import trange

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

def parameters(argv=None):
    parser = global_parameters()

    parser.add_argument('--model-path', default='./models/best.pth', type=str, help='Path to the model to inference')

    args = parser.parse_args(argv)
    args.add_noise = True if args.add_noise is not None else False
    args.use_simplified = True if args.use_simplified is not None else False
    return args

def measure_time(args, dataset, model):
    # Run the demo
    all_times = []
    for i in trange(1, dataset.__len__()):
        # Get an item and move tensors to correct device
        target, source, gt = dataset.__getitem__(i)
        target = target.unsqueeze(0).to(args.device)
        source = source.unsqueeze(0).to(args.device)

        # Run inference
        n_repetitions = 100
        times = timeit.timeit("model(source, target)",
                              globals={"model": model, "source": source, "target": target},
                              number=n_repetitions) / float(n_repetitions)
        all_times.append(times)

    average_time = sum(all_times) / float(len(all_times))
    log_message = "Average time per object: {:0.4f}ms".format(average_time * 10e3)
    LOGGER.info(log_message)

def load_model(args):
    # Load the model parameters
    tifreg_model = TIFRegSimplified(args) if args.use_simplified else TIFReg(args)
    checkpoint = torch.load(args.model_path, weights_only=False, map_location=args.device)
    tifreg_model.load_state_dict(checkpoint['model'])
    tifreg_model.to(args.device)
    tifreg_model.eval()

    for param in tifreg_model.parameters():  # Disable gradients to preserve memory
        param.requires_grad = False

    return tifreg_model


if __name__ == '__main__':
    logging_path = "./logs"
    if not os.path.exists(logging_path):
        os.mkdir(logging_path)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=f"{logging_path}/timing.log")

    options = parameters()

    if not torch.cuda.is_available():
        options.device = 'cpu'
    options.device = torch.device(options.device)

    # Create the dataset
    warnings.warn("Ignoring dataset option, using ModelNet40")
    chosen_dataset = ModelNet40(options, 'test')

    LOGGER.info("Original model")
    options.use_simplified = False
    tifreg_model = load_model(options)
    measure_time(options, chosen_dataset, tifreg_model)

    LOGGER.info("Simplified model")
    options.use_simplified = True
    tifreg_model = load_model(options)
    measure_time(options, chosen_dataset, tifreg_model)


