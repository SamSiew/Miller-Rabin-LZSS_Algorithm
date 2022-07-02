"""
Name: Ming Shern, Siew
studentid: 28098552
"""
class BinarySearchTreeNode:

    def __init__(self, key, value=None, left=None, right=None):
        self.key = key
        self.value = value
        self.left = left
        self.right = right

    def __getitem__(self, key):
        if self.left != None and self.left.key == key:
            return self.left
        if self.right != None and self.right.key == key:
            return self.right

    def add(self, new_key, val = None):
        if new_key == 1:
            if self.right == None:
                self.right = BinarySearchTreeNode(new_key, val)
            else:
                self.right = self.insert(self.right,new_key, val)
        else:
            if self.left == None:
                self.left = BinarySearchTreeNode(new_key, val)
            else:
                self.left = self.insert(self.left, new_key, val)

    def insert(self, current, new_key, value):
        if current == None:
            current = BinarySearchTreeNode(new_key, value)
        else:
            if new_key < current.key:
                current.left = self.insert(current.left, new_key, value)
            else:
                current.right = self.insert(current.right, new_key, value)
        return current

    def __str__(self):
        return "(" + str(self.key) + ", " + str(self.value) + ")"


class BinarySearchTree:

    def __init__(self):
        self.root = BinarySearchTreeNode(None)

    def is_empty(self):
        return self.root == None

    def __setitem__(self, key, value):
        self.root = self.insert(self.root, key, value)

    def add(self, new_key, val = None):
        if new_key == 1:
            if self.root.right == None:
                self.root.right = BinarySearchTreeNode(new_key, val)
            else:
                self.root.right = self.insert(self.root.right,new_key, val)
        else:
            if self.root.left == None:
                self.root.left = BinarySearchTreeNode(new_key, val)
            else:
                self.root.left = self.insert(self.root.left, new_key, val)

    def insert(self, current, new_key, value):
        if current == None:
            current = BinarySearchTreeNode(new_key, value)
        else:
            if new_key < current.key:
                current.left = self.insert(current.left, new_key, value)
            else:
                current.right = self.insert(current.right, new_key, value)
        return current

    def print_preorder(self):
        self.print_preorder_aux(self.root)

    def print_preorder_aux(self, current):
        if current != None:
            print(current, end=", ")
            self.print_preorder_aux(current.left)
            self.print_preorder_aux(current.right)

    def __getitem__(self, key):
        if self.root.left != None and self.root.left.key == key:
            return self.root.left
        if self.root.right != None and self.root.right.key == key:
            return self.root.right

    def search(self, key, current):
        if current.key == key:
            return current
        elif key < current.key:
            return self.search(key, current.left)
        else:
            return self.search(key, current.right)





