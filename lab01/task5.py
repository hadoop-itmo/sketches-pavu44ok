
import csv
from collections import defaultdict
import utils

KEY_THRESHOLD = 60000

def count_key_occurrences(file_path, key_column=0):
    key_occurrences = defaultdict(int)

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for record in csv_reader:
            key = record[key_column]
            key_occurrences[key] += 1

    return key_occurrences

def find_keys_w_problem(counts_file1, counts_file2):
    problematic_keys = set()

    for key, count in counts_file1.items():
        if count > KEY_THRESHOLD or counts_file2.get(key, 0) > KEY_THRESHOLD:
            problematic_keys.add(key)

    for key, count in counts_file2.items():
        if key not in counts_file1 and count > KEY_THRESHOLD:
            problematic_keys.add(key)

    return problematic_keys

def main(file1_path, file2_path):
    counts_file1 = count_key_occurrences(file1_path)
    counts_file2 = count_key_occurrences(file2_path)
    problem_keys = find_keys_w_problem(counts_file1, counts_file2)

    print(f'Проблемные ключи: {len(problem_keys)}')

    with open('problem_keys.txt', 'w') as output_file:
        for key in problem_keys:
            output_file.write(f'{key}\n')

def generate_test_data():
    pattern = [(10, 100_000)]
    utils.gen_grouped_seq('file1.csv', pattern, n_extra_cols=0, to_shuffle=False)
    utils.gen_grouped_seq('file2.csv', pattern, n_extra_cols=0, to_shuffle=False)

if __name__ == '__main__':
    generate_test_data()

    input_file1 = 'file1.csv'
    input_file2 = 'file2.csv'

    main(input_file1, input_file2)
