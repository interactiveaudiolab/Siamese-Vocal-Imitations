import argparse
import logging

import utils.network
import utils.utils as utilities
from data_files.vocal_sketch import VocalSketch_1_1
from data_sets.pair import AllPairs
from models.siamese import Siamese
from data_partitions.partitions import Partitions
from utils.inference import reciprocal_ranks
from data_partitions import PartitionSplit


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
    utils.network.load_model(model, cli_args.model_path, use_cuda=use_cuda)

    model = model.eval()
    data = VocalSketch_1_1()

    partitions = Partitions(data, PartitionSplit(.35, .15, .5))

    dataset = AllPairs(partitions.test)
    rrs = reciprocal_ranks(model, dataset, use_cuda)
    utilities.log_final_stats(rrs)


if __name__ == "__main__":
    main()
