import os

import numpy as np

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
