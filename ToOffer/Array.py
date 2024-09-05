"""
题目：
在一个长度为n的数组里有所有数字都在0~n-1的范围内，数组中某些数字是重复的，但不知道有几个数字重复了，
也不知道每个数字重复了几次，请找出数组中任意一个重复的数字，例如，如果输入长度为7的数组 [ 2, 3, 1, 0, 2, 5, 3 ] ，
那么对应的输出是重复的数字2或者3。

对原数组进行排序然后顺序查找，时间 O(nlogn) 空间 O(1)
利用哈希表解决，无需修改原数组，时间 O(n) 空间 O(n)
交换原数组中的元素，时间 O(n) 空间 O(1)
第三种方法最优，以下是实现
"""


# place num in the position num, if equal: dup, else: swipe with current
def duplicate(nums):
    if not nums:
        return nums
    id = 0
    for id in range(len(nums)):
        num = nums[id]
        if num >= len(nums):
            return "out of range"
        while id != num:
            if nums[id] == nums[num]:
                return num
            else:
                nums[id], nums[num] = nums[num], nums[id]
                num = nums[id]
                if num >= len(nums):
                    return "out of range"
    return "no dup"


# 测试用例
# 长度为 n 的数组里包含一个或多个重复的数字
test_case1 = [2, 3, 1, 0, 2, 5, 3]

# 数组中不含重复的数字
test_case2 = [2, 3, 1, 5, 4, 0]
# 无效输入测试用例
test_case3 = [1, 2, 3, 3]
test_case4 = [2, 6, 1, 0]
print("test case1:", duplicate(test_case1))
print("test case2:", duplicate(test_case2))
print("test case3:", duplicate(test_case3))
print("test case4:", duplicate(test_case4))

'''
题目：
在一个长度为n+1的数组里的所有数字都在1-n的范围内，所以数组中至少有一个数字是重复的，请找出数组中任意一个重复的数字，
但不能修改输入的数组，例如，如果输入长度为8 的数组 [ 2, 3, 5, 4, 3, 2, 6, 7 ] 那么对应的输出是重复的数字2或者3。
'''


# 找到当前数组的middle， 如果小于middle的数量>middle,则重复的在前面，否则在后面.front 考虑到middle，back从middle+1往后
# count考虑闭区间
# 空间O(1)， time O(nlogn)
def duplicate2(arr):
    length = len(arr)
    start = 1
    end = length - 1
    while start <= end:
        middle = (start + end) // 2
        count = count_number(arr, start, middle)
        if start == end:
            if count > 1: return start
        if count > (middle - start + 1):
            end = middle
        else:
            start = middle + 1


def count_number(arr, start, end):
    count = 0
    for i in range(len(arr)):
        if start <= arr[i] <= end:
            count += 1
    return count


# 测试用例
# 长度为n的数组里包含一个或多个重复的数字
test_case1 = [2, 3, 1, 4, 3, 2, 5, 7]
# 数组中不包含重复的数字
test_case2 = [1, 2, 3, 4, 5, 6, 7, 8]
print("test_case1:", duplicate2(test_case1))
print("test_case2:", duplicate2(test_case2))

'''
题目：
在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序，请完成一个函数，
输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

解题思路：
首先选取数组中右上角的数字。如果该数字。如果该数字等于要查找的数字，则查找过程结束；如果该数字大于要查找的数字，则剔除
这个数字所在的列；如果该数字小于要查找的数字，则剔除这个数字所在的行。也就是说，如果要查找的数字不在数组的右上角，则每
一次都在数组的查找范围中剔除一行或者一列，这样每一步都可以缩小查找的范围，直到找到要查找的数字，或者查找范围为空。
'''


def searchMatrix(matrix, target):
    row = 0
    col = len(matrix[0]) - 1
    while row < len(matrix) and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    return False


M = [[1, 4, 7, 11, 15],
     [2, 5, 8, 12, 19],
     [3, 6, 9, 16, 22],
     [10, 13, 14, 17, 24],
     [18, 21, 23, 26, 30]
     ]
print(searchMatrix(M, 9))
