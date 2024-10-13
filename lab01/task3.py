import mmh3
import pandas as pd
import numpy as np
from tqdm import tqdm
from common import generate_random_string


class CountingBloomFilter:
    def __init__(self, num_hashes, size, capacity=1):
        self.num_hashes = num_hashes
        self.size = size
        self.capacity = capacity
        self.total_size = size * capacity
        self.counters = np.zeros((self.total_size,), dtype=np.uint8)

    def put(self, item):
        for i in range(self.num_hashes):
            hash_index = mmh3.hash(item, i) % self.size
            self.counters[hash_index] = min(
                self.counters[hash_index] + 1,
                (1 << self.capacity) - 1,
            )

    def get(self, item):
        return all(self.counters[mmh3.hash(item, i) % self.size] > 0 for i in range(self.num_hashes))

    def remove(self, item):
        for i in range(self.num_hashes):
            hash_index = mmh3.hash(item, i) % self.size
            if self.counters[hash_index] > 0:
                self.counters[hash_index] -= 1

    def average_usage(self):
        return np.sum(self.counters) / self.num_hashes


if __name__ == '__main__':
    filter_sizes = [8, 64, 1024, 64 * 1024]
    data_set_sizes = [5, 50, 500, 5000, 5000000]
    hash_function_counts = [1, 2, 3, 4]

    test_results = []
    for size in tqdm(filter_sizes):
        for data_size in data_set_sizes:
            for num_hashes in hash_function_counts:
                counting_filter = CountingBloomFilter(num_hashes, size, capacity=5)
                unique_items = {generate_random_string() for _ in range(data_size)}
                true_positive_count = 0

                for item in unique_items:
                    counting_filter.put(item)

                for item in unique_items:
                    if counting_filter.get(item):
                        true_positive_count += 1

                average_counter = counting_filter.average_usage()
                test_results.append({
                    'filter_size': size,
                    'data_set_size': data_size,
                    'num_hashes': num_hashes,
                    'false_positive_count': true_positive_count,
                    'average_counter': average_counter
                })

    results_dataframe = pd.DataFrame(test_results)
    print('Тестирование задачи 3')
    print(results_dataframe)
