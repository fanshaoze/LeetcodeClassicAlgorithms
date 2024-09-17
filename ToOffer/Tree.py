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
        # root is first of the inorder, index is the root of the preorder,
        # from 1 to index+1 for inorder and preorder is left, from index+1 to end for inorder and preorder is right
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


# class _TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.parent = None

class _TreeNode:
    def __init__(self, x, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right


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


'''
题目：请完成一个函数，输入一颗二叉树，该函数输出它的镜像。
'''


def mirror_recursively(root):
    if root is None:
        return
    if root.left is None and root.right is None:
        return
    root.left, root.right = root.right, root.left
    if root.left is not None:
        mirror_recursively(root.left)
    if root.right is not None:
        mirror_recursively(root.right)


'''
题目：
输入数字n，按顺序打印出从1到最大的n位十进制数。比如输入3，则打印出1、2、3一直到最大的3位数999。
# 递归往后打印
'''


def numbersByRecursion(n):
    if n <= 0:
        return
    number = [0] * n
    print_to_max_of_n_digits_recursively(number, n, 0)


def print_number(number):
    is_beginning_0 = True
    for i in range(len(number)):  # print from the first that is not 0
        if is_beginning_0 and number[i] != '0':
            is_beginning_0 = False
        if not is_beginning_0:
            print(number[i], end='')
    print('\t')


def print_to_max_of_n_digits_recursively(number, length, index):
    if index == length:
        print_number(number)
        return
    for i in range(10):
        number[index] = str(i)
        print_to_max_of_n_digits_recursively(number, length, index + 1)


numbersByRecursion(3)

'''
题目：从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
'''


def layer(root):
    queue = [root]
    while queue:
        node = queue.pop(0)
        print(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)


def layer_print(root):
    current_queue = [root]
    while current_queue:
        queue = current_queue
        current_queue = []
        while queue:
            node = queue.pop(0)
            print(node.val)
            if node.left:
                current_queue.append(node.left)
            if node.right:
                current_queue.append(node.right)
        print("\n\n")


'''
题目：输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则返回true,否则返回false。假设输入的数组的任意两个数字都互不相同。
例如，输入数组{5,7,6,9,11,10,8},则返回true，因为这个整数序列是二叉搜索树的后序遍历结果。如果输入的数组是{7,4,6,5,}，则由于没有哪颗二叉搜索树的
后序遍历结果是这个序列，因此返回false。
'''

"""
最后一个一定是root，在某个位置id，0:id都小于root, id+1:-1都大于root, 否则wrong
"""


def verify_squence_of_bst(nums):
    if not nums or len(nums) == 1:
        return True
    root = nums[-1]
    idx = len(nums) - 1
    while idx >= 0 and nums[idx] >= root:
        idx -= 1
    split_id = idx
    for idx in range(split_id, -1, -1):
        if nums[idx] >= root:
            return False
    left = verify_squence_of_bst(nums[0:split_id + 1])
    right = verify_squence_of_bst(nums[split_id + 1:len(nums) - 1])
    return left and right


print(verify_squence_of_bst([5, 7, 6, 9, 11, 10, 8]))
print(verify_squence_of_bst([7, 4, 6, 5]))

'''
题目：输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
'''
"""
all path: if sum=exp_sum: add path
if <: path+root,
"""


def find_path(root, expected_sum):
    if not root:
        return []
    all_path = []

    def find_path_core(root, path, current_sum):
        print(root.val)
        if current_sum + root.val > expected_sum:
            return
        path.append(root.val)
        if current_sum + root.val == expected_sum:
            all_path.append(path)
        if root.left:
            find_path_core(root.left, path, current_sum + root.val)
        if root.right:
            find_path_core(root.right, path, current_sum + root.val)

    find_path_core(root, [], 0)
    return all_path


l = _TreeNode(5)
r = _TreeNode(7)
c0 = _TreeNode(6, l, r)
l = _TreeNode(9)
r = _TreeNode(11)
c1 = _TreeNode(10, l, r)
root = _TreeNode(8, c0, c1)

print(find_path(root, 14))

'''
Convert a Binary Search Tree to a sorted Circular Doubly-Linked List in place.
'''


def treeToDoublyList(root):
    if not root:
        return
    dummy = _TreeNode(0, None, None)
    stack = []
    pre = dummy
    node = root
    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        # update double link
        pre.right = node
        node.left = pre
        pre = node

        node = node.right
    # update dummy
    dummy.right.left = pre
    pre.right = dummy.right
    return dummy.right


'''
题目：
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
'''


def more_than_half_num(numbers):
    if not numbers:
        return
    key, count = None, 0
    for num in numbers:
        if key is None:
            key, count = num, 1
        else:
            if key == num:
                count += 1
            else:
                count -= 1
        if count == 0:
            key = None
    return key


'''
题目：输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
'''
import heapq


def get_least_numbers(nums, k):
    heaps = []
    i = 0
    for n in nums:
        if i < k:
            heapq.heappush(heaps, -n)
            i += 1
        else:
            heapq.heappushpop(heaps, -n)
    return [-i for i in heaps]


print(get_least_numbers([1, 2, 3, 4, 5, 5, 6, 7], 3))

'''
题目：输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。要求时间复杂度为O(n)。
'''


def max_subarray(nums):
    pre_sum = 0
    min_pre = 0
    max_sub = 0
    for n in nums:
        pre_sum += n
        max_sub = max(max_sub, pre_sum - min_pre)
        min_pre = min(min_pre, pre_sum)
    print(max_sub)


max_subarray([1, -2, 3, 10, -4, 7, 2, -5])

'''
题目：如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
'''
"""
维护两个堆，mid前的最大堆和mid后的最小堆，最大堆多存一个，每有一个新数据，push到数量少的一个堆，并对比堆顶，如果反了做交换
"""

import heapq


class med:
    def __init__(self):
        self.max_smaller = []
        self.min_larger = []

    def insert(self, num):
        if not num:
            return None
        else:
            if len(self.max_smaller) <= len(self.min_larger):
                heapq.heappush(self.max_smaller, -num)
            else:
                heapq.heappush(self.min_larger, num)

            if (-self.max_smaller[0]) > self.min_larger[0]:
                heapq.heappush(self.max_smaller, -heapq.heappop(self.min_larger))
                heapq.heappush(self.min_larger, -heapq.heappop(self.max_smaller))

    def get_med(self):
        if len(self.min_larger) == len(self.max_smaller):
            return (-heapq.heappop(self.max_smaller) + heapq.heappop(self.min_larger)) / 2
        else:
            return -heapq.heappop(self.max_smaller)


'''
题目：输入一个整数n，求1~n这n个整数的十进制表示中1出现的次数。例如，输入12，1~12这些整数中包含1的数字有1、10、11和12，1一共出现了5次。
'''

"""
get before, current and after: pre, mod = divmod(nums, unit * 10), current_number, after = divmod(mod, unit)
再组合
"""


def number_of_1(nums):
    unit = 1
    count = 0
    while nums // unit > 0:
        pre, mod = divmod(nums, unit * 10)
        current_number, after = divmod(mod, unit)
        if current_number > 1:
            count += pre * unit + unit
        elif current_number == 1:
            count += pre * unit + after + 1
        else:
            count += pre * unit
        unit *= 10
    return count


'''
题目：数字以0123456789101112131415……的格式序列化到一个字符串序列中。在这个序列中，第5位（从0开始计数）是5，第13位是1，第19位是4，等等。
请写一个函数，求任意第n位对应的数字。
'''

"""
用位数确定所在数字，按位数逐渐跳过小于位数的所有值，再反向生成
"""


def digit_at_index(index):
    if index < 0:
        return None
    digits = 1
    while True:
        numbers = 10 if digits == 1 else 9 * 10 ** (digits - 1)
        if index < numbers * digits:
            number = (10 ** (digits - 1) if digits != 1 else 0) + index // digits
            right_index = digits - index % digits
            for i in range(0, right_index - 1):
                number = number // 10
            return number % 10
        index -= digits * numbers
        digits += 1


print(digit_at_index(18))
print(digit_at_index(19))
print(digit_at_index(20))

from functools import cmp_to_key

"""
@param nums: n non-negative integer array
@return: A string
"""


def min_number(self, nums):
    # write your code here
    if not nums:
        return
    key = cmp_to_key(lambda x, y: int(x + y) - int(y + x))
    res = ''.join(sorted(map(str, nums), key=key)).lstrip('0')
    return res or '0'


'''
题目：给定一个数字，我们按照如下规则把它翻译为字符串：0翻译成“a“，1翻译成”b“，……，……25翻译成”z“。一个数字可能有多少个翻译。例如，12258有5种
不同的翻译，分别是“bccfi”、“bwfi”、“bczi“、”mcfi“和”mzi“。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
'''


def translation(nums):
    counts = [0 for i in range(len(nums) + 1)]

    for i in range(len(nums) - 1, -1, -1):
        if i != len(nums) - 1:
            count = counts[i + 1]
            double_num = int(nums[i]) * 10 + int(nums[i + 1])
            if double_num < 26:
                count += counts[i + 2]
        else:
            count = 1
        counts[i] = count

    return counts[0]


print(translation('12258'))

'''
题目：在一个mxn的期盼的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于0。你可以从棋盘的左上角开始拿格子里的礼物，并每次向左或向下移动一格，
直到到达棋盘的右下角。给定要给棋盘及其上面的礼物，请计算你最多能拿到多少价值的礼物？
'''
"""
递归计算到current position最大值，先往右再往下
"""

def get_max_value(values, rows, cols):
    if not values or rows <= 0 or cols <= 0:
        return 0
    max_values = [[0] * cols] * rows
    for i in range(rows):
        for j in range(cols):
            left = 0
            up = 0
            if i > 0:
                up = max_values[i - 1][j]
            if j > 0:
                left = max_values[i][j - 1]
            max_values[i][j] = max(up, left) + values[i][j]
    return max_values[rows - 1][cols - 1]


print(get_max_value([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3, 3))


'''
题目：请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。假设字符串中只包含‘a’~‘z’的字符。例如，在字符串”arabcacfr“中，
最长的不含重复字符的子字符串是“acfr”，长度为4。
'''

"""
record position to avoid while loop popping
"""
def longest_substring_without_duplication(str):
    if not str:
        return
    string = list(str)
    current_length = 0
    max_length = 0
    position = [0] * 26
    for i in range(len(string)):
        prev_index = position[ord(string[i]) - ord('a')]
        if i - prev_index > current_length:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = i - prev_index  # count from previous string[i+1]
        position[ord(string[i]) - ord('a')] = i
    return max_length

print(longest_substring_without_duplication('arabcacfr'))



'''
题目：输入一棵二叉树的根结点，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
'''


class Solution:
    def tree_depth(self, root):
        if not root:
            return 0
        return max(self.tree_depth(root.left), self.tree_depth(root.right)) + 1

'''
题目：输入两个树结点，求它们的最低公共祖先。
'''


def get_last_common_parent(root, node1, node2):
    if root is None:
        return None
    if root == node1 or root == node2:
        return root
    left_result = get_last_common_parent(root.left, node1, node2)
    right_result = get_last_common_parent(root.right, node1, node2)
    # A 和 B 一边一个
    if left_result and right_result:
        return root
    # 左子树有一个点或者左子树有LCA
    if left_result:
        return left_result
    # 右子树有一个点或者右子树有LCA
    if right_result:
        return right_result
    # 左右子树都没有
    return None

# Finish
