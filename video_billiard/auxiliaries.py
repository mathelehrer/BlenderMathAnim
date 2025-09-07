import random

selected_primes = []

def random_prime(seed=None,**kwargs):
    """Returns a random prime number.

    Args:
        seed (int): Random seed for reproducibility

    Returns:
        int: A random prime number

    Examples:
        >>> random_prime(seed=1234)
        97
    """

    if seed is not None:
        random.seed(seed)
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
    sel_prime = random.choice(primes)
    while sel_prime in selected_primes and len(selected_primes)<len(primes):
        sel_prime = random.choice(primes)
    selected_primes.append(sel_prime)
    return sel_prime

if __name__ == '__main__':
    print(random_prime())