# Function to search a given key in a given BST
class BSTNode:
    def __init__(self,key):
        self.key = key
        self.left = None
        self.right = None

    def __str__(self):
        return str(self.key)

    def __repr__(self):
        return repr(self.key)

# A utility function to insert
# a new node with the given key in BST
def insert(node,key):
    # if the group_tree is empty, return a new node
    if node is None:
        return BSTNode(key),True

    # otherwise, recur down the group_tree
    if key<node.key:
        node.left, success = insert(node.left,key)
    elif key>node.key:
        node.right, success = insert(node.right,key)
    else:
        success = False

    return node, success

# utility function to search a key in a BST
def search(root,key):
    # Base cases: root is null or key is present at root
    if root is None or root.key==key:
        return root

    # key is greater than root's key
    if root.key<key:
        return search(root.right,key)
    return search(root.left,key)

if __name__ == '__main__':
    root = None
    root,success = insert(root,50)
    print("Root: ",root)
    print("30: ",insert(root,30))
    print("20: ",insert(root,20))
    print("20: ",insert(root,20))
    print("40: ",insert(root,40))
    print("70: ",insert(root,70))
    print("60: ",insert(root,60))
    print("60: ",insert(root,60))
    print("50: ",insert(root,50))
    print("80: ",insert(root,80))

    key =6
    #search in a BST
    if search(root,key) is None:
        print(key,"not found")
    else:
        print(key,"found")

    key = 60
    # search in a BST
    if search(root, key) is None:
        print(key, "not found")
    else:
        print(key, "found")

    root2 = None
    root2, success = insert(root2, -6646856180955596176)
    print("-7580525578129651975: ", insert(root2, -7580525578129651975))
    print("3159689373340834797: ", insert(root2, 3159689373340834797))
    print("-6425840997236830215: ", insert(root2, -6425840997236830215))
    print("3858666435499615037: ", insert(root2, 3858666435499615037))
    print("-2994507527303831279: ", insert(root2, -2994507527303831279))
    print("1428576568674662871: ", insert(root2, 1428576568674662871))
    print("-2934580132080335655: ", insert(root2, -2934580132080335655))
    print("-4730232155723714842: ", insert(root2, -4730232155723714842))
    print("-4036511917941570261: ", insert(root2, -4036511917941570261))
    print("-6425840997236830215: ", insert(root2, -6425840997236830215))