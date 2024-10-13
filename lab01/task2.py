import mmh3
import pandas as pd
import numpy as np
from tqdm import tqdm
from common import generate_random_string


class AdvancedBloomFilter:
    def __init__(self, k, size):
        self.k = k
        self.size = size
        self.bit_vector = np.zeros((self.size,), dtype=bool)

    def put(self, item):
        for i in range(self.k):
            hash_index = mmh3.hash(item, i) % self.size
            self.bit_vector[hash_index] = True

    def get(self, item):
        return all(self.bit_vector[mmh3.hash(item, i) % self.size] for i in range(self.k))

    def average_ones_per_k(self):
        return np.sum(self.bit_vector) / self.k


if __name__ == '__main__':
    bloom_filter_sizes = [8, 64, 1024, 64 * 1024, 16 * 1024 * 1024]
    data_set_sizes = [5, 50, 500, 5000, 5000000]
    hash_function_counts = [1, 2, 3, 4]

    test_results = []
    for filter_size in tqdm(bloom_filter_sizes):
        for data_size in data_set_sizes:
            for hash_count in hash_function_counts:
                bloom_filter_instance = AdvancedBloomFilter(hash_count, filter_size)
                unique_items = {generate_random_string() for _ in range(data_size)}
                true_positive_count = 0

                for item in unique_items:
                    if bloom_filter_instance.get(item):
                        true_positive_count += 1
                    bloom_filter_instance.put(item)

                average_ones = bloom_filter_instance.average_ones_per_k()
                test_results.append({
                    'filter_size': filter_size,
                    'data_set_size': data_size,
                    'hash_count': hash_count,
                    'false_positive_count': true_positive_count,
                    'average_ones': average_ones
                })

    results_dataframe = pd.DataFrame(test_results)
    print('Тестирование задачи 2')
    print(results_dataframe)
