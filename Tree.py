class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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


TreeNode(1, None, None)


def LCA(root, p, q):
    if root is None or p == root or q == root:  # if p is root of q, p is the LCA, vice versa.
        return root  # if p and not find in the brother of p, than must under p, so no need for searching on l and r
    else:
        l = LCA(root.left, p, q)
        r = LCA(root.right, p, q)
    if l and r:  # both exist in l and r, means l and r in different sub-tree seperately
        return root
    else:
        if l:
            return l
        else:
            return r


def LCA_count(root, p, q, count):
    if root is None:
        return None, count
    else:
        l, count = LCA_count(root.left, p, q, count)
        r, count = LCA_count(root.right, p, q, count)

        if root.val == p.val or root.val == q.val:
            return root, count + 1
        elif l and r:
            return root, count
        elif l:
            return l, count
        elif r:
            return r, count
        else:
            return None, count


def LCA_depth(root):
    if root is None:
        return root, -1
    else:
        l, dl = LCA_depth(root.left)
        r, dr = LCA_depth(root.right)
        # print(root.val, dl, dr)
        if l is None and r is None:
            return root, 0
        elif dl > dr:
            return l, dl + 1
        elif dr > dl:
            return r, dr + 1
        else:
            return root, dl + 1  # or dr+1


def preorderTraversal(root, returns):
    if root is None:
        return
    else:
        returns.append(root.val)
        preorderTraversal(root.left, returns)
        preorderTraversal(root.right, returns)
    return returns

def preorderTraversal_stack(root):
    if not root:
        return []

    stack = [root]
    result = []

    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result


def inorderTraversal_stack(root):
    result = []
    stack = []
    current = root

    while current or stack:
        while current:
            stack.append(current)
            current = current.left

        current = stack.pop()
        result.append(current.val)
        current = current.right  # use none to control the pop

    return result




def postorderTraversal_stack(root):
    if not root:
        return []

    stack = []
    result = []
    prev = None

    while root or stack:
        while root:
            stack.append(root)
            root = root.left

        root = stack[-1]

        if not root.right or root.right == prev:
            result.append(root.val)
            stack.pop()
            prev = root
            root = None
        else:
            root = root.right

    return result


def BFS(root):
    if root is None:
        return
    res = []
    q = [root]
    while q:
        level = []
        for i in range(len(q)):  # the len(q) is the length of current level
            temp = q.pop(0)
            level.append(temp.val)
            if temp.left is not None:
                q.append(temp.left)
            if temp.right is not None:
                q.append(temp.right)
        res.append(level)
    return res


def find_path_to_root(root, target):
    if root is None:
        return None
    if root.val == target.val:
        return [root]
    left_path = find_path_to_root(root.left, target)
    if left_path:
        return [root] + left_path
    right_path = find_path_to_root(root.right, target)
    if right_path:
        return [root] + right_path
    return None


def main():
    tree = Tree([i for i in range(7)])
    root = tree.root
    print(preorderTraversal(root, []))
    print(inorderTraversal_stack(root))
    print(postorderTraversal_stack(root))
    print(find_path_to_root(root, root.left.left.right))
    print(LCA(root, root.left.right, root.left.left.left).val)
    print(BFS(root))


if __name__ == '__main__':
    main()
