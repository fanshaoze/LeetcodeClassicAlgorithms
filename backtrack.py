def numIslands(grid):
    m, n = len(grid), len(grid[0])
    total = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == "1":
                total += 1
                visit_count(grid, i, j)
    return total


def visit_count(grid, i, j):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == '0':
        return
    grid[i][j] = "0"
    visit_count(grid, i + 1, j)
    visit_count(grid, i, j + 1)
    visit_count(grid, i - 1, j)
    visit_count(grid, i, j - 1)


print(numIslands([["1", "1", "1", "1", "0"],
                  ["1", "1", "0", "0", "0"],
                  ["1", "1", "0", "1", "0"],
                  ["0", "0", "0", "0", "1"]]))
