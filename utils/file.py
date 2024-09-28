import os


def read_lines(filename):
    file = open(filename, 'r')
    file_content = file.read()
    words = file_content.splitlines()
    file.close()

    return words


def load_array_from_file(filename):
    words = read_lines(filename)

    word_set = set(words)

    assert len(words) == len(word_set)

    return words


def create_folder_if_not_exist(folder_name):
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
