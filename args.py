import argparse

# Global parameters for all files
# Files should call this function and add their own specific args
def global_parameters(argv=None):
    parser = argparse.ArgumentParser(description='TIF Registration')

    # Deep Feature Embedding (DGCNN) Parameters
    parser.add_argument('--k', default=20, type=int,
                        metavar='N', help='Number of (k) neighbours to use in DGCNN')
    parser.add_argument('--dropout', default=0.5, type=float,
                        metavar='N', help='Probability to use in DGCNNs dropout')
    parser.add_argument('--use-simplified', action='store', nargs='*', help='Uses the simplified model over the original')


    # Dataset Parameters
    parser.add_argument('--dataset', default='modelnet', choices=['modelnet', 'bunny', 'tum3d', 'mould', 'mould_random'], type=str,
                        help="Name of the dataset to use") # TODO Merge random dataset classes and just change the input target path
    parser.add_argument('--num-points', default=2048, type=int,
                        metavar='N', help='Number of points to use in ModelNet40')
    parser.add_argument('--max-rot', default=90, type=float,
                        metavar='T', help='Maximum rotation in degrees. Applies in range [-x, +x] (default: 90)')
    parser.add_argument('--max-trans', default=0.5, type=float,
                        help='Maximum translation in each axis (default: 0.5)')
    parser.add_argument('--add-noise', action='store', nargs='*', help='Whether to add noise to the source')

    # Utility settings
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available', choices=['cuda', 'cuda:0', 'cpu'])

    return parser