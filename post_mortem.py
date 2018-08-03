import os
import argparse
import subprocess

import utils.post_mortem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=str, help="Output directory where the trial output folders can be found")
    parser.add_argument('-rt', "--representative_trial", type=int, help="Representative trial number, used to generate loss/rank overlay graph.")
    parser.add_argument("-d", '--download', action="store_const", default=False, const=True,
                        help="Download latest results from Cortex (run ./download_output). Defaults to false.")
    parser.add_argument("-v", '--verbose', action="store_const", default=False, const=True,
                        help="Be verbose. Defaults to false")

    cli_args = parser.parse_args()

    output_dir = os.path.abspath(cli_args.output_directory)
    representative_trial = cli_args.representative_trial
    download = cli_args.download
    verbose = cli_args.verbose

    if download:
        print("Refreshing trial output from Cortex...")
        subprocess.run("./download_output", stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)
        print("Done.")

    utils.post_mortem.condense_graphs(output_dir, verbose=verbose)
    correlations = utils.post_mortem.get_correlations(output_dir)
    diff, p_value = utils.post_mortem.wilcox_test(correlations)
    print("Average correlation difference of {0} is statistically significant at a p-value = {1}".format(diff, p_value))
    utils.post_mortem.correlation_csv(output_dir, correlations)
    utils.post_mortem.boxplot(output_dir, correlations, p_value)
    if not representative_trial:
        representative_trial = utils.post_mortem.get_representative_trial(correlations)
        utils.post_mortem.loss_rank_overlay(output_dir, representative_trial)


if __name__ == "__main__":
    main()
