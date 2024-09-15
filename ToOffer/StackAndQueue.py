'''
题目：用两个栈来实现一个队列，完成队列的Push、Pop、Top操作。
'''


class MyQueue:
    def _init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, val):
        self.stack1.append(val)

    def pop(self):
        if self.stack2:
            self.stack2.pop()
        elif not self.stack1:
            return False
        else:
            while self.stack1:
                val = self.stack1.pop()
                self.stack2.append(val)
            self.stack2.pop()


"""
通过pop，每次都取最上，左，下，右的list，删除防止重复访问
"""


def spiralOrder(matrix):
    result = []
    while matrix:
        if matrix[0]:
            result += matrix[0]
            matrix.pop(0)
        if matrix:
            for line in matrix:
                result += [line.pop()]
        if matrix:
            last_line = matrix.pop()
            last_line.reverse()
            result += last_line
        if matrix:
            for line in matrix:
                result += [line.pop(0)]
    print(result)


spiralOrder([[1, 2, 3, 4],
             [1, 2, 3, 4],
             [1, 2, 3, 4]])

'''
题目：定义栈的数据结构，请在该类型中实现一个能够得到栈最小元素的min函数。在该栈中，调用min、push及pop的时间复杂度都是O(1)。
'''

"""
每次入/出栈，在最少栈也入/出一个当前最小
"""


class MinStack:

    def __init__(self):
        self.data_stack = []
        self.min_stack = []

    def push(self, number):
        self.data_stack.append(number)
        if len(self.min_stack) == 0 or number < self.min_stack[-1]:
            self.min_stack.append(number)
        else:
            self.min_stack.append(self.min_stack[-1])

    def pop(self):
        if len(self.data_stack) == 0 and len(self.min_stack) == 0:
            return None
        if len(self.data_stack) > 0 and len(self.min_stack) > 0:
            self.min_stack.pop()
            return self.data_stack.pop()

    def min(self):
        # write your code here
        return self.min_stack[-1]


'''
题目：输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列{1,2，3，4,5}是
某栈的压栈序列，序列{4,5,3,2,1}是该压栈序列对应的一个弹出序列，但{4,3,5,1,2}就不可能是该压栈序列的弹出序列。
'''


def is_pop_order(p_push, p_pop):
    stack = []
    while p_pop:
        if stack and stack[-1] == p_pop[0]:  # current should pop
            stack.pop()
            p_pop.pop(0)
        elif p_push:  # future to pop, push to stack
            stack.append(p_push.pop(0))
        else:
            return False
    return True


print(is_pop_order([1, 2, 3, 4, 5], [4, 5, 3, 2, 1]))
