import os
import numpy as np


def scan_files_with_sizes_recursive(directory, max_number=np.Inf):
    results = {}
    for root, dirs, files in os.walk(directory):
        for name in files:
            file = os.path.join(root, name)
            try:
                size = os.path.getsize(file)
            except OSError:
                continue
            results[file] = size
            if len(results) > max_number:
                return results
    return results


if __name__ == '__main__':
    directory = '/usr'
    results = scan_files_with_sizes_recursive(directory, max_number=1000)
    digits = {}
    total = 0
    for size in results.values():
        d = int(str(size)[0])
        if d > 0:
            digits.update({d: digits.get(d, 0) + 1})
            total += 1
    for d in range(1, 10):
        print(f"{d}: {digits[d] / total:.2f} <-> {np.log((d + 1) / d) / np.log(10):.2f}")
