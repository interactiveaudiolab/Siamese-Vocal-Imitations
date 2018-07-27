import os
import shutil
import sys


def main():
    if len(sys.argv) < 1:
        raise ValueError("Provide all arguments: condense_graphs.py [directory]")

    d = sys.argv[1]

    for roots, dirs, files in os.walk(d):
        for file in files:
            if file.endswith("png"):
                src = os.path.join(roots, file)
                p = os.path.join(os.path.dirname(roots), os.path.basename(roots) + '_' + file)
                print("{0} --> {1}".format(src, p))
                shutil.copyfile(src, p)


if __name__ == "__main__":
    main()
