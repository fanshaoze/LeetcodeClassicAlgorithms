'''
题目：
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
'''

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


class Solution:
    """
    @param inorder: A list of integers that inorder traversal of a tree
    @param postorder: A list of integers that postorder traversal of a tree
    @return: Root of a tree
    """

    def buildTree(self, inorder, preorder):
        # write your code here
        if not preorder or not inorder:
            return  #

        root = TreeNode(inorder[0])
        index = preorder.index(inorder[0])

        root.left = self.buildTree(inorder[1:index + 1], preorder[:index])
        root.right = self.buildTree(inorder[index + 1:], preorder[index + 1:])

        return root


'''
题目：
给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，
同时包含指向父结点的指针。
'''

'''
如果有右子树，则找右子树的最左leaf
否则，找是由哪个节点往左搜索得到的该节点，所以如果父节点的右子树不是当前节点，说明找到了
'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.parent = None


def next_node(pNode):
    if not pNode:
        return None
    if pNode.right is not None:
        pNode = pNode.right
        while pNode.left is not None:
            pNode = pNode.left
        return pNode
    elif pNode.parent is not None:
        while pNode.parent is not None and pNode.parent.right == pNode:
            pNode = pNode.parent
        return pNode.parent
