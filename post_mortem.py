import os
import sys

from utils.post_mortem import condense_graphs, generate_correlation_csv


def main():
    if len(sys.argv) <= 1:
        raise ValueError("Provide an output directory (python post_mortem.py output_dir).")

    output_dir = sys.argv[1]
    output_dir = os.path.abspath(output_dir)

    condense_graphs(output_dir)
    generate_correlation_csv(output_dir)


if __name__ == "__main__":
    main()
