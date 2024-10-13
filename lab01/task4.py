import mmh3
import pandas as pd
import numpy as np
from tqdm import tqdm
from common import generate_random_string


class HyperLogLog:
    def __init__(self, precision):
        self.precision = precision
        self.num_buckets = 1 << precision
        self.alphaMM = (0.7213 / (1 + 1.079 / self.num_buckets)) * self.num_buckets ** 2
        self.registers = np.zeros(self.num_buckets, dtype=int)

    def put(self, item):
        x = mmh3.hash(item)
        bucket_index = x & (self.num_buckets - 1)
        self.registers[bucket_index] = max(self.registers[bucket_index], self._rho(x))

    def _rho(self, hash_value):
        hash_value = hash_value >> 32
        return (hash_value | 1).bit_length()

    def estimate_size(self):
        harmonic_mean = 1.0 / np.sum([2.0 ** -reg for reg in self.registers])
        estimate = self.alphaMM * harmonic_mean

        if estimate <= (5.0 / 2.0) * self.num_buckets:
            zeros_count = np.count_nonzero(self.registers == 0)
            if zeros_count > 0:
                estimate = self.num_buckets * np.log(self.num_buckets / zeros_count)
        elif estimate > (1 << 32):
               estimate = -(2 ** 32) * np.log(1 - estimate / (2 ** 32))

        return int(estimate)


if __name__ == '__main__':
    precision_values = [4, 6, 8, 10]
    num_tests = 100
    unique_counts = [100, 1000, 10000, 100000]

    task4_results = []
    for precision in tqdm(precision_values):
        for unique_count in unique_counts:
            hll = HyperLogLog(precision)
            unique_strings = {generate_random_string() for _ in range(unique_count)}

            # Add strings to HyperLogLog
            for s in unique_strings:
                hll.put(s)

            estimated_count = hll.estimate_size()

            task4_results.append({
                'precision': precision,
                'unique_count': unique_count,
                'estimated_count': estimated_count,
                'actual_count': len(unique_strings),
                'error': abs(estimated_count - len(unique_strings))  # Calculate error
            })


    task4_df = pd.DataFrame(task4_results)
    print('Тестирование задачи 4')
    print(task4_df)

    error_summary = task4_df.groupby('precision')['error'].mean().reset_index()
    print(error_summary)