'''
题目：
输入一个链表的头节点，从尾到头反过来打印出每个节点的值。
'''


def PrintListReversingly(head):
    if head:
        if head.next:
            PrintListReversingly(head.next)
        print(head.val)


# 栈
def PrintListReversingly2(head):
    stack = []
    while head is not None:
        stack.append(head.val)
        head = head.next
    while stack:
        print(stack.pop())
    print()
