import mmh3
import pandas as pd
import numpy as np
from tqdm import tqdm
from common import generate_random_string


class BloomFilter:
    def __init__(self, size):
        self.size = size
        self.bit_vector = np.zeros((self.size,), dtype=bool)

    def put(self, item):
        hash_index = mmh3.hash(item) % self.size
        self.bit_vector[hash_index] = True

    def get(self, item):
        hash_index = mmh3.hash(item) % self.size
        return self.bit_vector[hash_index]

    def count_ones(self):
        return np.sum(self.bit_vector)


if __name__ == '__main__':
    bloom_filter_sizes = [8, 64, 1024, 64 * 1024, 16 * 1024 * 1024]
    data_set_sizes = [5, 50, 500, 5000, 5000000]

    test_results = []
    for filter_size in tqdm(bloom_filter_sizes):
        for data_size in data_set_sizes:
            bloom_filter = BloomFilter(filter_size)
            unique_items = {generate_random_string() for _ in range(data_size)}
            true_positive_count = 0

            for item in unique_items:
                if bloom_filter.get(item):
                    true_positive_count += 1
                bloom_filter.put(item)

            total_ones = bloom_filter.count_ones()
            test_results.append({
                'filter_size': filter_size,
                'data_set_size': data_size,
                'false_positive_count': true_positive_count,
                'total_ones': total_ones
            })

    results_dataframe = pd.DataFrame(test_results)
    print('Тестирование задачи 1')
    print(results_dataframe)
