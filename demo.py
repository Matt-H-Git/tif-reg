import argparse

from args import global_parameters
from datasets.tum3d import TUM3D
from utils import *
import time
import torch.utils.data

from datasets.modelnet40 import ModelNet40
from datasets.stanford_bunny import StanfordBunny
from tif.model import TIFReg, TIFRegSimplified

def parameters(argv=None):
    parser = global_parameters()

    parser.add_argument('--model-path', default='./models/best.pth', type=str, help='Path to the model to inference')

    args = parser.parse_args(argv)
    args.add_noise = True if args.add_noise is not None else False
    args.use_simplified = True if args.use_simplified is not None else False
    return args

if __name__ == '__main__':
    args = parameters()

    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    # Load the model parameters from somewhere
    if args.use_simplified:
        model = TIFRegSimplified(args)
    else:
        model = TIFReg(args)
    checkpoint = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()

    for param in model.parameters(): # Disable gradients to preserve memory
        param.requires_grad = False

    # Create the dataset
    n_examples = 100
    dataset = None
    if args.dataset == "modelnet":
        dataset = ModelNet40(args, 'test')
    if args.dataset == "bunny":
        dataset = StanfordBunny(args, "./data/stanford_bunny/stanford_bunny.ply", n_elements=n_examples, n_points=2048)
    if args.dataset == "tum3d":
        dataset = TUM3D(args, 'test')


    # Run the demo
    for i in range(1, dataset.__len__()):
        # Get an item and move tensors to correct device
        target, source, gt = dataset.__getitem__(i)
        target = target.unsqueeze(0).to(args.device)
        source = source.unsqueeze(0).to(args.device)
        source_original = source.clone()
        target_original = target.clone()

        # Run inference
        t1 = time.time_ns()
        R_est, t_est = model(source, target)
        t2 = time.time_ns()
        print("Time to inference Pytorch model: ", str((t2 - t1) / 1e6), "ms")

        # Convert result to 4x4 transform
        gT_est = matrix_from_R_and_t(R_est, t_est).to(args.device)

        # Print and visualise results
        print("Estimate transform")
        print(gT_est.cpu().detach().t().numpy())
        print("Actual transform")
        print(gt.cpu().detach().t().numpy())
        print("Loss: ", torch.nn.functional.mse_loss(gT_est.cpu(), gt.cpu()))

        #print("Drawing registration using ground truth transform")
        #draw_registration_result(source_original, target_original, gt)

        print("Drawing registration using estimated transform")
        draw_registration_result(source_original, target_original, gT_est)


