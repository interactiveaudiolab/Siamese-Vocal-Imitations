import argparse
import logging

import utils.utils as utilities
from data_files.vocal_sketch import VocalSketchV2
from data_sets.siamese import AllPairs
from models.siamese import Siamese
from data_partitions.siamese import SiamesePartitions
from utils.experimentation import reciprocal_ranks
from utils.obj import DataSplit


def main():
    logger = logging.getLogger('logger')
    utilities.configure_logger(logger, console_only=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help="Path to the model to be evaluated")
    parser.add_argument('-c', '--cuda', action='store_const', const=True, default=False,
                        help="Whether to enable calculation on the GPU through CUDA or not. Defaults to false.")

    cli_args = parser.parse_args()
    use_cuda = cli_args.cuda

    model = Siamese(dropout=False)
    if use_cuda:
        model.cuda()
    utilities.load_model(model, cli_args.model_path, use_cuda=use_cuda)

    model = model.eval()
    data = VocalSketchV2()

    partitions = SiamesePartitions(data, DataSplit(.35, .15, .5))

    dataset = AllPairs(partitions.test)
    rrs = reciprocal_ranks(model, dataset, use_cuda)
    utilities.log_final_stats(rrs)


if __name__ == "__main__":
    main()
