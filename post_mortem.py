import os
import argparse
import subprocess

import utils.post_mortem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=str, help="Output directory where the trial output folders can be found")
    parser.add_argument("-d", '--download', action="store_const", default=False, const=True,
                        help="Download latest results from Cortex (run ./download_output). Defaults to false.")
    parser.add_argument("representative_trial", type=int, help="Representative trial number, used to generate loss/rank overlay graph.")

    cli_args = parser.parse_args()
    download = cli_args.download

    if download:
        subprocess.run("./download_output")

    output_dir = os.path.abspath(cli_args.output_directory)
    representative_trial = cli_args.representative_trial

    utils.post_mortem.condense_graphs(output_dir)
    correlations = utils.post_mortem.get_correlations(output_dir)
    utils.post_mortem.generate_correlation_csv(output_dir, correlations)
    utils.post_mortem.generate_boxplot(output_dir, correlations)
    utils.post_mortem.overlay_loss_rank(output_dir, representative_trial)


if __name__ == "__main__":
    main()
