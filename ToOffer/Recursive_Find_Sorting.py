def fibonacci(self, n):
    # write your code here
    result = [0, 1]
    for i in range(2, n + 1):
        result.append(result[i - 1] + result[i - 2])
        return result[n]


# https://leetcode.com/problems/word-search/description/

board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
word = "ABCCED"


def exist(board, word):
    def search(board, i, j, word):
        print(i, j, m, n, word)
        if i < 0 or i >= m or j < 0 or j >= n:
            return False
        elif len(word) == 0:
            return True
        else:
            if board[i][j] != word[0]:
                return False
            tmp = board[i][j]
            board[i][j] = '.'
            a = search(board, i + 1, j, word[1:])
            b = search(board, i, j + 1, word[1:])
            c = search(board, i - 1, j, word[1:])
            d = search(board, i, j - 1, word[1:])
            board[i][j] = tmp
            return a or b or c or d

    m, n = len(board), len(board[0])
    for i in range(m):
        for j in range(n):
            a = search(board, i, j, word)
            if a: return True
    return False


print(exist(board, word))


def findminInRotated(nums):
    left, right = 0, len(nums) - 1
    ret = nums[0]
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:  # not sorted, must on right side
            left = mid + 1
        else:
            ret = nums[mid]
            right = mid - 1
    return ret


print(findminInRotated([3, 4, 5, 1, 2]))


def max_product_after_cutting_solution2(length):
    if length < 2:
        return 0
    if length == 2:
        return 1
    if length == 3:
        return 2
    times_of_3 = length // 3
    if length - times_of_3 * 3 == 1:  # split 3+1 to 2+2
        times_of_3 -= 1
    times_of_2 = (length - times_of_3 * 3) / 2
    return (3 ** times_of_3) * (2 ** times_of_2)


''''
题目：
地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，
但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。
但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
'''


def moving_count(threshold, rows, cols):
    if threshold < 0 or rows <= 0 or cols <= 0:
        return 0
    visited = [False for i in range(rows * cols)]
    count = moving_count_core(threshold, rows, cols, 0, 0, visited)
    return count


def moving_count_core(threshold, rows, cols, row, col, visited):
    count = 0
    if check(threshold, rows, cols, row, col, visited):
        visited[row * cols + col] = True
        count = 1 + moving_count_core(threshold, rows, cols, row - 1, col, visited) + moving_count_core(
            threshold, rows, cols, row + 1, col, visited) + moving_count_core(threshold, rows, cols, row,
                                                                              col - 1,
                                                                              visited) + moving_count_core(
            threshold, rows, cols, row, col + 1, visited)
    return count


def check(threshold, rows, cols, row, col, visited):
    if 0 <= row < rows and 0 <= col < cols and get_digit_sum(row) + get_digit_sum(
            col) <= threshold and not visited[row * cols + col]:
        return True
    return False


def get_digit_sum(number):
    sum = 0
    while number > 0:
        sum += number % 10
        number //= 10
    return sum


moving_count(18, 10, 10)

'''
题目：
请实现一个函数，输入一个整数，输出该数二进制表示中1的个数。例如，把9表示成二进制是1001，有2位是1,。因此，如果输入9，
则该函数输出2。
举个例子：一个二进制数1100，从右边数起第三位是处于最右边的一个1。减去1后，第三位变成0，它后面的两位0变成了1，
而前面的1保持不变，因此得到的结果是1011.我们发现减1的结果是把最右边的一个1开始的所有位都取反了。这个时候如果我们
再把原来的整数和减去1之后的结果做与运算，从原来整数最右边一个1那一位开始所有位都会变成0。如1100&1011=1000.也就是
说，把一个整数减去1，再和原整数做与运算，会把该整数最右边的1变成0。那么一个整数的二进制表示中有多少个1，就可以进行多少次
这样的操作。
'''


def countOnes(num):
    count = 0
    if num < 0:
        num = num & 0xffffffff
    while num:
        count += 1
        num = (num - 1) & num
    return count


'''
题目：
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。不得使用库函数，同时不需要考虑最大数问题。
'''


def power_with_exponent(base, exponent):
    if exponent == 0:
        return 1
    if exponent == 1:
        return base
    result = power_with_exponent(base, exponent // 2)
    result *= result
    if exponent % 2 == 1:
        result *= base
    return result


print(power_with_exponent(2, 3))

'''
题目：输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
'''


def convert(self, root):
    # write your code here
    if not root:
        return None
    self.inorder = []
    self.inorder_traversal(root)
    for i in range(len(self.inorder) - 1):
        self.inorder[i].right = self.inorder[i + 1]
        self.inorder[i + 1].left = self.inorder[i]
    return self.inorder[0]


def inorder_traversal(self, root):
    if not root:
        return
    self.inorder_traversal(root.left)
    self.inorder.append(root)
    self.inorder_traversal(root.right)


'''
题目:输入一个字符串，打印出该字符串中字符的所有排列顺序。例如:输入字符串abc，则打印出字符a、b、c所能排列出的所有字符串abc、acb、bac、bca、
cab和cba。
'''


def permute2(nums):
    ret = []

    def add_perm(lst, cur):
        if not lst:  # reach last
            return
        if len(cur) == len(nums):
            ret.append(cur)
        else:
            for i in range(len(lst)):
                add_perm(lst[:i] + lst[i + 1:], cur + [lst[i]])

    add_perm(nums, [])
    return ret


'''
题目：我们把只包含因子2、3和5的数称作丑数（Ugly Number）。求按从小到大的顺序的第N个丑数。
例如6、8都是丑数，但14不是，因为它包含因子7。习惯上我们把1当做是第一个丑数。
'''
"""
maintain the next one that is larger than current, update
"""


def get_ugly(index):
    uglys = [1 for i in range(index)]
    t2_id, t3_id, t5_id = 0, 0, 0
    for i in range(1, index):
        uglys[i] = min(uglys[t2_id] * 2, uglys[t3_id] * 3, uglys[t5_id] * 5)
        while uglys[t2_id] * 2 <= uglys[i]:
            t2_id += 1
        while uglys[t3_id] * 3 <= uglys[i]:
            t3_id += 1
        while uglys[t5_id] * 5 <= uglys[i]:
            t5_id += 1
    print(uglys)
    return uglys[-1]


get_ugly(14)

'''
题目：输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，输出任意一对即可。
'''


def find_numbers_with_sum(data, sum):
    if not data:
        return
    r = len(data) - 1
    l = 0
    while l < r:
        cur_sum = data[l] + data[r]
        if cur_sum == sum:
            return l, r
        elif cur_sum > sum:
            r -= 1
        else:
            l += 1
    return False


'''
题目：在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组
中的逆序对的总数。例如，在数组{7,5,6,4}中，一共存在5个逆序对，分别是{7，6}、{7,5}、{7,4}、{6,4}和{5，4}。
'''


class Solution:
    def inverse_pairs(self, data):
        self.copy = [0] * len(data)
        return self.inverse_pairs_core(data, 0, len(data) - 1)

    def inverse_pairs_core(self, data, l, r):
        if l >= r:
            self.copy[l] = data[l]
            return 0
        middle = (l + r) // 2
        result = self.inverse_pairs_core(data, l, middle) + self.inverse_pairs_core(data, middle + 1, r)
        i, j, k = l, middle + 1, l
        while i <= middle and j <= r:
            if data[i] > data[j]:
                result += middle + 1 - i
                self.copy[k] = data[j]
                j += 1
            else:
                self.copy[k] = data[i]
                i += 1
            k += 1
        while i <= middle:
            self.copy[k] = data[i]
            i += 1
            k += 1

        while j <= r:
            self.copy[k] = data[j]
            j += 1
            k += 1

        for i in range(l, r + 1):
            data[i] = self.copy[i]
        print(data)
        return result


s = Solution()
print(s.inverse_pairs([7, 5, 6, 4]))

'''
题目：统计一个数字在排序数组中出现的次数。例如，输入排序数组{1,2,3,3,3,3,4,5,}和数字3，由于3在这个数组中出现了4次，因此输出4。
'''


def find_most_right(nums, n):
    res = -1
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] <= n:
            res = mid
            l = mid + 1
        else:
            r = mid - 1
    return res


def find_most_left(nums, n):
    res = -1
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] >= n:
            res = mid
            r = mid - 1
        else:
            l = mid + 1
    return res


def find_num_equal(nums, n):
    return find_most_right(nums, n) - find_most_left(nums, n) + 1


nums = [1, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8]
print(find_most_right(nums, 4))
print(find_most_left(nums, 4))

'''
题目:一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0~n-1之内。在范围0~n-1内的n个数字中有且只有一个数字不在该数组中，
请找出这个数字。
'''


def get_missing_number(nums):
    if not nums:
        return
    res = -1
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] != mid:
            res = mid
            end = mid - 1
        else:
            start = mid + 1
    return res


print(get_missing_number([0, 1, 2, 4, 5, 6, 7, 8]))


def get_number_same_as_index(nums):
    if not nums:
        return -1
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = (right + left) >> 1
        if nums[mid] == mid:
            return mid
        elif nums[mid] > mid:
            right = mid - 1
        elif nums[mid] < mid:
            left = mid + 1
    return -1


print(get_number_same_as_index([-3, -1, 1, 3, 5]))

'''
题目：给定一棵二叉搜索树，请找出其中的第k大的结点。
'''


class solution:
    def __init__(self, n):
        self.ret = None
        self.n = n

    def kth_node(self, root):
        self.kth_node_core(root, 0)

    def kth_node_core(self, root, k):
        if self.ret is not None:
            return
        if not root:
            return
        else:
            self.kth_node_core(root.left, k)
            if self.ret: return
            if k + 1 == self.n:
                self.ret = root
                return
            self.kth_node_core(root.right, self.n - (k + 1))


'''
题目：给定一个数组和滑动窗口的大小，请找出所有滑动窗口里的最大值。例如，如果输入数组{2, 3, 4, 2, 6, 2, 5, 1}及滑动窗口的大小3，
那么一共存在6个滑动窗口，它们的最大值分别为{4, 4, 6, 6, 6, 5}，
'''

import collections


def get_max_in_windows(nums, size):
    max_in_windows = []
    if nums and 1 <= size <= len(nums):
        index = collections.deque()
        for i in range(size):
            while index and nums[i] >= nums[index[-1]]:
                index.pop()
            index.append(i)
        for i in range(size, len(nums)):
            max_in_windows.append(nums[index[0]])
            while index and nums[i] >= nums[index[-1]]:  # we will not consider the last one before reaching nums[i]
                index.pop()
            if index and index[0] <= i - size:  # window remove a largest
                index.popleft()
            index.append(i)
        max_in_windows.append(nums[index.popleft()])  # the last one
    else:
        return []
    return max_in_windows


print(get_max_in_windows([2, 3, 4, 2, 6, 2, 5, 1], 3))

'''
题目：把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
'''


def print_probability(num):
    probabilities = [[0 for i in range(6 * num + 1)] for i in range(num)]
    for i in range(6):
        probabilities[0][i] = 1
    for i in range(1, num):
        for j in range(i, 6 * (i + 1)):  # [0,i-1]的时候，频数为0（例如2个骰子不可能投出点数和为1）
            probabilities[i][j] = probabilities[i - 1][j - 6] + probabilities[i - 1][j - 5] + probabilities[i - 1][
                j - 4] + probabilities[i - 1][j - 3] + probabilities[i - 1][j - 2] + probabilities[i - 1][j - 1]
    result_probability = probabilities[num - 1]
    total = 6 ** num
    for s in range(num, 6 * num + 1):
        print(s, result_probability[s - 1] / total)


'''
题目：从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王可以看成任意数字。
'''


def is_continuous(nums):
    if not nums or len(nums) != 5:
        return False
    nums = sorted(nums)
    # 统计数组中0的个数
    num_of_zero = len([i for i in nums if i == 0])
    num_of_gap = 0
    # 统计数组中的间隔数目
    small = num_of_zero
    big = small + 1
    while big < len(nums):
        if nums[small] == nums[big]:
            return False
        num_of_gap += nums[big] - nums[small] - 1
        small += 1
        big += 1
    return False if num_of_gap > num_of_zero else True


'''
题目：0, 1, …, n-1这n个数字排成一个圆圈，从数字0开始每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。
'''


def last_remaining(n, m):
    if n < 1 or m < 1:
        return -1
    last = 0
    for i in range(2, n + 1):
        last = (last + m) % i
    return last


'''
题目：假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖交易该股票可能获得的利润是多少？例如一只股票在某些时间节点的价格为{9, 11, 8, 5,
7, 12, 16, 14}。如果我们能在价格为5的时候买入并在价格为16时卖出，则能收获最大的利润11。
'''

"""
pre-min
"""


def max_diff(nums):
    if not nums and len(nums) < 2:
        return 0
    min = nums[0]
    max_diff = nums[1] - min
    for i in range(2, len(nums)):
        if nums[i - 1] < min:
            min = nums[i - 1]
        cur_diff = nums[i] - min
        if cur_diff > max_diff:
            max_diff = cur_diff
    return max_diff


'''
题目：求1+2+…+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

方法二：利用两个函数，一个函数充当递归函数的角色，另一个函数处理终止递归的情况。如果对n连续进行两次反运算，
那么非零的n转换为True，0转换为False。利用这一特性终止递归。注意考虑测试用例为0的情况。
'''
class Solution2:
    def sum_solution(self, n):
        return self.sum(n)

    def sum0(self, n):
        return 0

    def sum(self, n):
        fun = {False: self.sum0, True: self.sum}
        return n + fun[not not n](n - 1)


'''
题目：给定一个数组A[0, 1, …, n-1]，请构建一个数组B[0, 1, …, n-1]，其中B中的元素B[i] =A[0]×A[1]×… ×A[i-1]×A[i+1]×…×A[n-1]。
不能使用除法。
'''

def multiply(a):
    if not a:
        return
    length = len(a)
    b = [1] * length
    for i in range(1, length):
        b[i] = b[i - 1] * a[i - 1]
    tmp = 1
    for i in range(length - 2, -1, -1):
        tmp *= a[i + 1]
        b[i] *= tmp
    return b
