import random
import string


def generate_random_string(length=10):
    return ''.join(
        random.choice(string.ascii_lowercase)
        for _ in range(length)
    )
