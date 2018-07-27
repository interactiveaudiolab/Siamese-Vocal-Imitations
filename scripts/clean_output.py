import os
import shutil


def main():
    d = './output'
    for roots, dirs, files in os.walk(d):
        if files == ['siamese.log']:
            print(roots)
            shutil.rmtree(roots)


if __name__ == "__main__":
    main()
