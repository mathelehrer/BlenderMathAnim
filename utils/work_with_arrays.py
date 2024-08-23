import numpy as np

array = np.array([[col*row for col in range(1,5) ]for row in range(7,10)])
diag = ((0,0),(1,1),(2,2))
print(array)
print(array[tuple(np.array(diag).T)])

# https://stackoverflow.com/questions/28491230/indexing-a-numpy-array-with-a-list-of-tuples

cube =np.array([[0,0,0.],[0,0,1.],[0,1,0.],[0,1,1.],[1,0,0.],[1,0,1.],[1,1,0.],[1,1,1.]])
print(cube)

# select three points, whose indices are given as tuples

print([cube[idx] for idx in (1,2,3)])
