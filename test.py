import random


class Tree(object):
    def __init__(self, vals):
        root = TreeNode(vals[0], None, None)
        self.root = root
        for val in vals[1:]:
            node = TreeNode(val, None, None)
            if root.left is None:
                root.left = node
            elif root.right is None:
                root.right = node
                root = root.left


class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def BFS(root):
    node = root
    queue = [node]
    while queue:
        node = queue.pop(0)
        print(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)


def preorder_stack(root):
    node = root
    stack = []
    while node or stack:
        while node:
            stack.append(node)
            print(node.val)
            node = node.left
        node = stack.pop()
        if root.right:
            node = node.right
        else:
            node = None


def inorder_stack(root):
    node = root
    stack = []
    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        print(node.val)
        if node.right:
            node = node.right
        else:
            node = None


def post_order_stack(root):
    if not root:
        return []
    stack = []
    prev = None
    node = root

    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack[-1]
        if not node.right or prev == node.right:
            print(node.val)
            prev = node
            stack.pop()
            node = None
        else:
            node = node.right


l = TreeNode(5)
r = TreeNode(7)
c0 = TreeNode(6, l, r)
l = TreeNode(9)
r = TreeNode(11)
c1 = TreeNode(10, l, r)
root = TreeNode(8, c0, c1)
BFS(root)
print(" ")
preorder_stack(root)
print(" ")
inorder_stack(root)
print(" ")
post_order_stack(root)


def twosum(nums, target):
    remaining = {}
    res = []
    for idx, n in enumerate(nums):
        if n not in remaining:
            remaining[target - n] = idx
        else:
            res.append([nums[remaining[n]], n])
    return res


def k_sum(nums, target, k):
    res = []
    if not nums:
        return res
    if k == 2:
        return twosum(nums, target)
    else:
        for i, n in enumerate(nums):
            for subset in k_sum(nums[i + 1:], target - n, k - 1):
                res.append([n] + subset)
    return res


print(twosum([0, 1, 2, 3, 4, 5, 6, 7, 8], 8))
print(k_sum([0, 1, 2, 3, 4, 5, 6, 7, 8], 16, 4))
