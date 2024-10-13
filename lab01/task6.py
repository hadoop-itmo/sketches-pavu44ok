import mmh3
import numpy as np
import utils
import shutil


class BloomFilter:
    def __init__(self, filter_size, hash_function_count):
        self.filter_size = filter_size
        self.hash_function_count = hash_function_count
        self.bit_array = np.zeros(filter_size, dtype=bool)

    def _hash_functions(self, item):
        return [mmh3.hash(item, i) % self.filter_size for i in range(self.hash_function_count)]

    def add(self, item):
        for h in self._hash_functions(item):
            self.bit_array[h] = True

    def contains(self, item):
        return all(self.bit_array[h] for h in self._hash_functions(item))


class CountMinSketch:
    def __init__(self, table_width, table_depth):
        self.table_width = table_width
        self.table_depth = table_depth
        self.frequency_table = np.zeros((table_depth, table_width), dtype=int)
        self.hash_functions = [lambda x, i=i: mmh3.hash(x, i) % table_width for i in range(table_depth)]

    def add(self, item):
        for i, h in enumerate(self.hash_functions):
            self.frequency_table[i][h(item)] += 1

    def estimate(self, item):
        return min(
            self.frequency_table[i][h(item)]
            for i, h in enumerate(self.hash_functions)
        )


def read_file_w_filter(file_path, bloom_filter, count_min_sketch, max_unique_keys):
    unique_keys = set()
    total_keys = 0

    with open(file_path, 'r') as file:
        for line in file:
            key = line.split(',')[0]
            if key not in unique_keys and total_keys < max_unique_keys:
                unique_keys.add(key)
            bloom_filter.add(key)
            count_min_sketch.add(key)
            total_keys += 1

    return unique_keys


def estimate(file_path, bloom_filter, count_min_sketch):
    estimated_size = 0

    with open(file_path, 'r') as file:
        for line in file:
            key = line.split(',')[0]
            if bloom_filter.contains(key):
                estimated_size += count_min_sketch.estimate(key)

    return estimated_size


def main():
    pattern = [(10, 100_000)]
    file1 = "test_data1.csv"
    file2 = "test_data2.csv"
    utils.gen_grouped_seq(file1, pattern, n_extra_cols=0, to_shuffle=True)
    shutil.copyfile(file1, file2)

    bloom_filter_capacity = 10 ** 7
    hash_functions_count = 7
    sketch_table_width = 10 ** 5
    sketch_table_depth = 5
    max_unique_key_limit = 10 ** 6

    bloom_filter1 = BloomFilter(bloom_filter_capacity, hash_functions_count)
    sketch1 = CountMinSketch(sketch_table_width, sketch_table_depth)

    unique_keys_in_first_file = read_file_w_filter(file1, bloom_filter1, sketch1, max_unique_key_limit)

    if len(unique_keys_in_first_file) <= max_unique_key_limit:
        bloom_filter2 = BloomFilter(bloom_filter_capacity, hash_functions_count)
        sketch2 = CountMinSketch(sketch_table_width, sketch_table_depth)
        unique_keys_in_second_file = read_file_w_filter(file2, bloom_filter2, sketch2, max_unique_key_limit)
        intersection_keys = unique_keys_in_first_file & unique_keys_in_second_file
        print(f"Количество пересечений: {len(intersection_keys)}")
    else:
        estimated_join_size = estimate(file2, bloom_filter1, sketch1)
        print(f"Оценка количества пересечений: {estimated_join_size}.")

if __name__ == '__main__':
    main()
