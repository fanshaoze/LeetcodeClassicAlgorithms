'''
题目：请实现一个函数，将一个字符串中的空格替换成“%20”。
例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
'''

"""
从前往后每次添加需要挪动后续string，实际运行开销为O(n^2)
从后往前，先确定多出来的长度，之后设置快慢针，如果慢针不是空格则直接替换掉快针位置，如果是空格，把快针往前三个格设置为%20
"""


def replacement(string):
    string = list(string)
    count = 0
    for s in string:
        if s == ' ': count += 1
    l = len(string) - 1
    l_new = len(string) + count * 2 - 1
    string += [' '] * count * 2
    print(string)
    while l >= 0:
        if string[l] != ' ':
            string[l_new] = string[l]
            l -= 1
            l_new -= 1
        else:
            string[l_new - 2] = '%'
            string[l_new - 1] = '2'
            string[l_new] = '0'
            l_new -= 3
            l -= 1
    print(''.join(string))


replacement('123 456 789')

'''
题目:
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
'''
"""
可能形式：
AeEB, A, B 都可以是数或者小数，B不能是空 
"""


def is_value(string):
    string = list(string)
    e_id = None
    e_count = 0
    for idx in range(len(string)):
        if string[idx] == 'e':
            if not e_id: e_id = idx
            e_count += 1
    if e_count > 1:
        return False
    elif e_count == 1:
        pre = string[0:e_id]
        post = string[e_id + 1:]
        if not post:
            return False
        else:
            return is_number(pre) and is_number(post)

    else:
        return is_number(string)


def is_number(string):
    digit = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    signal = ['+', '-']
    dot = 0
    for idx in range(len(string)):
        if string[idx] in signal:
            if idx == 0:
                continue
            else:
                return False
        if string[idx] == '.':
            if dot != 0:
                return False
            else:
                dot = 1
                continue
        else:
            if string[idx] not in digit:
                return False
            else:
                continue
    return True


print(is_value('123e123'))
print(is_value('-1.23e+12.3'))
print(is_value('-1.23e'))
print(is_value('-1.23e2e'))

'''
题目：
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
'''
"""
Partition
"""


def reorder_odd_even(nums):
    l = 0
    r = len(nums) - 1
    while l < r:
        while nums[l] % 2 == 1 and l < r:
            l += 1
        while nums[r] % 2 == 0 and l < r:
            r -= 1
        if l < r:
            nums[l], nums[r] = nums[r], nums[l]
    return nums


print(reorder_odd_even([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
