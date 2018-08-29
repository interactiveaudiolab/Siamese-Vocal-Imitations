import os


def recursive_wav_paths(path):
    """
    Get the paths of all .wav files found recursively in the path.

    :param path: path to search recursively in
    :return: list of absolute paths
    """
    absolute_paths = []
    for folder, subs, files in os.walk(path):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension.lower() == '.wav':
                file_path = os.path.join(folder, file)
                absolute_paths.append(os.path.abspath(file_path))

    return absolute_paths
