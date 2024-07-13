import numpy as np

class Node:
    def __init__(self, id, value=None, b=None):
        """
        :param value: int
        """
        self.id = id
        self.value = value
        self.b = b
        self.left = None
        self.right = None

class Tree:
    def __init__(self, max_depth):
        """
        Generate a binary tree in array representation with a given depth
        :param max_depth: int
        """
        totalNodes = int(np.power(2, max_depth + 1) - 1)  # 2^(max_depth) - 1
        divide = int(np.ceil(totalNodes / 2))

        self.max_depth = max_depth
        self.branch_nodes = np.arange(1, divide)
        self.leaf_nodes = np.arange(divide, totalNodes + 1)

    def tree_from_array(self, array, predictions=None, a = None, b = None):
        """
        :return: Node
        """
        if predictions is None:
            predictions = []
        if a is None:
            a = []
        if b is None:
            b = []


        new_array = []
        for i in range(len(array)):
            element = {
                'id': array[i],
                'value': predictions[i] if i < len(predictions) else None,
                'b': b[array[i]].value if array[i] in b else None
            }
            new_array.append(element)                    

        return self._build_tree(new_array, None, 0, len(array))
    
    def _build_tree(self, arr, root, i, n):
        if i < n:
            temp = Node(arr[i]['id'], arr[i]['value'], arr[i]['b'])
            root = temp

            # Insert left child
            root.left = self._build_tree(arr, root.left, 2 * i + 1, n)

            # Insert right child
            root.right = self._build_tree(arr, root.right, 2 * i + 2, n)
        return root
    
    def print_tree(self, predictions=None, a = None, b = None):
        """
        Print the tree
        """
        root = self.tree_from_array(np.concatenate([self.branch_nodes, self.leaf_nodes]), predictions, a, b)
        self._print_tree(root)

    def _print_tree(self, root, level=0, prefix="Root: "):
        if root is not None:
            print(" " * (level * 4) + prefix + f"({root.id}) val: " + str(root.value) + f" b: {root.b}")
            if root.left or root.right:
                if root.left:
                    self._print_tree(root.left, level + 1, "L--- ")
                else:
                    print(" " * ((level + 1) * 4) + "L---" + f"({root.id})val: " + root.value + f"b: {root.b}")

                if root.right:
                    self._print_tree(root.right, level + 1, "R--- ")
                else:
                    print(" " * ((level + 1) * 4) + "R---" + f"({root.id})val: " + root.value + f"b: {root.b}")


    def right_ancestors(self, node_index):
        """
        :param node_index: int
        :return: np.array
        """
        if (node_index not in self.branch_nodes) and (node_index not in self.leaf_nodes):
            raise ValueError(f'Node "{node_index}" is not in the tree')

        ancestors = []
        current = node_index

        while current != 1:
            parent = current // 2

            if current % 2 == 1:
                ancestors.append(parent)

            current = parent

        return np.array(ancestors)

    def left_ancestors(self, node_index):
        """
        :param node_index: int
        :return: np.array
        """
        if (node_index not in self.branch_nodes) and (node_index not in self.leaf_nodes):
            raise ValueError(f'Node "{node_index}" is not in the tree')

        ancestors = []
        current = node_index

        while current != 1:
            parent = current // 2

            if current % 2 == 0:
                ancestors.append(parent)

            current = parent

        return np.array(ancestors)