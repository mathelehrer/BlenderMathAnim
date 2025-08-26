import os

import numpy as np

from utils.constants import DATA_DIR

def list_files_with_sizes_recursive(directory: str, follow_links: bool = False,max_number=np.Inf):
    results = {}
    for root, dirs, files in os.walk(directory, followlinks=follow_links):
        for name in files:
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            results[path]=size
            if len(results)>max_number:
                return results
    return results

def convert_files_to_csv_data(path,max_length,max_data,max_number=10000):
    results = list_files_with_sizes_recursive(path,follow_links=False, max_number=max_number)
    selected_results = {}
    for ext_file, size in results.items():
        file = ext_file.split("/")[-1]
        if size > 0 and len(file) <= max_length and '.' in file:
            selected_results[file] = size

    print(str(len(selected_results))+" files selected from directory")

    digits = {}
    for size in selected_results.values():
        d = int(str(size)[0])

        if d in digits:
            digits[d] += 1
        else:
            digits[d] = 1

    for d in range(1, 10):
        print(
            f"{d}: {digits[d] / (len(selected_results)):.2f} <-> {np.log(d + 1) / np.log(10) - np.log(d) / np.log(10):.2f}")

    print(len(selected_results), " data points.")

    # prepare data for output
    with open(os.path.join(DATA_DIR, path.replace("/","")+"_data.csv"), "w") as f:
        for i in range(max_length+2):
            f.write(str(i) + ",")
        f.write("size\n")
        count = 0
        for file, size in selected_results.items():
            for i in range(max_length):
                if len(file) > i:
                    f.write(str(ord(file[i])) + ",")
                else:
                    f.write("32" + ",")
            # two additional spaces
            f.write("32"+",")
            f.write("32"+",")
            f.write(str(size) + "\n")
            count += 1
            if count > max_data:
                break

