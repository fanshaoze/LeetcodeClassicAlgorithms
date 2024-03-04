def dfs():
    pass


def bfs():
    pass


def graph_row_column_dfs(grid, i, j, m, n):  # filled with '0' or '1'
    if i < 0 or j < 0 or i >= m or j >= n:
        return
    else:
        if grid[i][j] == '1':
            # print(i,j)
            grid[i][j] = '0'
            graph_row_column_dfs(grid, i - 1, j, m, n)  # pre may not be considered as search from dfs but not iteration
            graph_row_column_dfs(grid, i + 1, j, m, n)
            graph_row_column_dfs(grid, i, j - 1, m, n)
            graph_row_column_dfs(grid, i, j + 1, m, n)
        else:
            return


def dfs_search_path(i, j, board, m, n, word, idx):
    if i < 0 or i >= m or j < 0 or j >= n:
        return False
    else:
        if board[i][j] != word[idx]:
            return False
        else:
            tmp = board[i][j]
            board[i][j] = '.'
            if idx == len(word) - 1:
                return True
            else:
                t = dfs_search_path(i, j - 1, board, m, n, word, idx + 1)
                d = dfs_search_path(i, j + 1, board, m, n, word, idx + 1)
                l = dfs_search_path(i - 1, j, board, m, n, word, idx + 1)
                r = dfs_search_path(i + 1, j, board, m, n, word, idx + 1)
                board[i][j] = tmp
                if t or d or l or r:
                    return True


def dfs_count_sum(root, avg_c):
    if root is None:
        return 0, 0, avg_c
    else:
        l_sum, lc, avg_c = dfs_count_sum(root.left, avg_c)
        r_sum, rc, avg_c = dfs_count_sum(root.right, avg_c)
        sub_sum = l_sum + r_sum + root.val
        count = lc + rc + 1

        if sub_sum // count == root.val:
            avg_c += 1
        print(sub_sum, count, avg_c)
    return sub_sum, count, avg_c

lst = []
def dfs_bit(root, sum):

    if root is None:
        return
    elif root.left is None and root.right is None:
        lst.append(10*sum+root.val)
    else:
        cur = 10*sum+root.val
        dfs_bit(root.left, cur)
        dfs_bit(root.right, cur)
