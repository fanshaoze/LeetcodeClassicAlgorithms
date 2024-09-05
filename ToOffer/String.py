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
