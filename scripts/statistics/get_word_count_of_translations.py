import pandas as pd
import os
from glob import glob
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../../utils')))
from useful_variables import african_languages


def count_words_in_csv(filename):
    word_count = 0
    # Do not count previously-translated datasets (non-virology Afrikaans, Zulu, or Xhosa datasets)
    if any(suffix in os.path.basename(filename) for suffix in {'_af', '_zu', '_xh'}) and 'virology' not in os.path.basename(filename):
        return word_count
    try:
        df = pd.read_csv(filename, encoding='utf-8', header=(None if 'mmlu' in filename else 'infer'))
        if 'mmlu' in filename:
            for column in df.columns[:-1]:
                word_count += df[column].astype(str).apply(lambda x: len(x.split())).sum()
        else:
            for column in df.columns[4:7]:
                word_count += df[column].astype(str).apply(lambda x: len(x.split())).sum()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    return word_count


def count_words_in_files(file_list):
    total_word_count = 0
    for file in file_list:
        if os.path.isfile(file):
            word_count = count_words_in_csv(file)
            print(f"File: {os.path.basename(file)} - Words: {word_count}")
            total_word_count += word_count
        else:
            print(f"File not found: {os.path.basename(file)}")
    print(f"Total word count: {total_word_count}")
    return total_word_count


if __name__ == "__main__":
    file_list = []
    for lang in african_languages:
        file_list.extend(sorted(glob(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../data/evaluation_benchmarks_afr_release/mmlu_cm_ck_vir/*_{lang}.csv')))))
        file_list.extend(sorted(glob(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), f'../../data/evaluation_benchmarks_afr_release/winogrande_s/*_{lang}.csv')))))
    output_path = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__), '../../results/statistics/word_count_of_translations.txt'))
    with open(output_path, 'w') as f:
        # Redirect sys.stdout to the file
        sys.stdout = f
        count_words_in_files(file_list)
