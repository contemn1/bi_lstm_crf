import logging
import re
import json
from os import listdir
from os.path import join
from  sklearn.model_selection import train_test_split

pattern = '[\[\]=]'
regex = re.compile(pattern)

def extract_string(file_path):
    try:
        with open(file_path) as input_file:
            input_strings = input_file.readlines()
            output_list_temp = list()
            output_dict = []
            for input_string in input_strings:
                input_string = input_string.strip('\n')
                input_string = regex.sub('', input_string)
                input_string = input_string.strip()
                if input_string:
                    input_string = input_string.split(' ')
                    for input_string_split in input_string:
                        output_list_temp.append(input_string_split)
            index = 0
            for word in output_list_temp:
                word_list = word.split('/')
                if len(word_list) == 2:
                    output_dict.append(' '.join(word_list))
                    if word_list[0] == '.':
                        output_dict.append('')
                index += 1

            return output_dict

    except IOError as err:
        logging.error('Failed to open file {0}'.format(err.message))


def output_file(file_name_list, output_path):
    with open(output_path, 'w+') as file1:
        for file in file_name_list:
            file1.write('\n'.join(extract_string(file)))


if __name__ == '__main__':
    root_path = '/Users/zxj/Dropbox/zxj_qiu/treebank_3/tagged/pos/wsj'
    sub_dirs = [join(root_path, dir) for dir in listdir(root_path)]
    files = [join(sub_dir, file) for sub_dir in sub_dirs for file in listdir(sub_dir) if '.pos' in file]
    part_files = files[:1000]
    other, test = train_test_split(part_files, test_size=0.1)
    train, valid = train_test_split(other, test_size=0.11)
    output_path_root = '../data/treebank3/en'
    output_file(train, join(output_path_root, 'train.txt'))
    output_file(test, join(output_path_root, 'test.txt'))
    output_file(valid, join(output_path_root, 'valid.txt'))
