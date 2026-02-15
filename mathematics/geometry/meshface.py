class MeshFace(list):
    """
        this is just a list of integers that is hashable, we want to make sure that
        the ording of the indices is preserved and cyclicly rotated to have the smallest index
        in the first position

        However, the reverse of the list should have the same hash value as the original list
    """

    def __init__(self, face_indices=[]):
        """
        hashable list that is equivalent with respect to cyclic permutations
        >>> MeshFace([1,2,3]) == MeshFace([3,1,2])
        True
        >>> MeshFace([1,2,3]) == MeshFace([1,3,2])
        True
        >>> MeshFace([1,2,4]) == MeshFace([1,2,3])
        False
        """
        super().__init__(face_indices)
        smallest_index = min(self)

        index = self.index(smallest_index)

        for i in range(index):
            face_indices.append(face_indices.pop(0))

        self.reverse_elements = face_indices.copy()
        self.reverse_elements.reverse()
        self.reverse_elements.insert(0, self.reverse_elements.pop())
        self.elements = face_indices

    def __str__(self):
        """
        >>> str(MeshFace([1,2,3]))
        'Face([1, 2, 3])'
        >>> str(MeshFace([2,3,1]))
        'Face([1, 2, 3])'
        >>> str(MeshFace([2,1,3]))
        'Face([1, 3, 2])'

        """
        return f"MeshFace("+str(self.elements)+")"

    def __repr__(self):
        return f"MeshFace("+str(self.elements)+")"

    def __hash__(self):
        """
        It is assumed that the size of the elements are less than 20000
        >>> hash(MeshFace([1,0,2]))
        9050618495337975385
        >>> hash(MeshFace([0,1,2]))
        9050618495337975385
        >>> hash(MeshFace([1,2,0]))
        9050618495337975385
        >>> hash(MeshFace([0,1,2,3]))
        -4042512932752085617
        >>> hash(MeshFace([3,2,1,0]))
        -4042512932752085617
        >>> hash(MeshFace([1,2,3,0]))
        -4042512932752085617
        """
        return hash(tuple(self.elements))*hash(tuple(self.reverse_elements))

    def __eq__(self,other):
        first = self.elements == other.elements
        second = self.elements == other.reverse_elements
        return first or second
