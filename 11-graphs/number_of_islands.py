"""
Number of Islands - LeetCode Problem 200
https://leetcode.com/problems/number-of-islands/

Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), 
return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally 
or vertically. You may assume all four edges of the grid are surrounded by water.

Example 1:
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1

Example 2:
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 300
- grid[i][j] is '0' or '1'.
"""

from typing import List
import time
from collections import deque


class Solution:
    """
    Solution class with multiple approaches to solve Number of Islands
    """
    
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        Main method - returns the most efficient solution
        """
        return self.num_islands_dfs(grid)
    
    def num_islands_dfs(self, grid: List[List[str]]) -> int:
        """
        Approach 1: Depth-First Search (DFS) - Most Common
        
        Algorithm:
        1. Iterate through each cell in the grid
        2. When we find a '1', start DFS to mark all connected land
        3. Mark visited cells as '0' to avoid revisiting
        4. Count the number of DFS calls (islands)
        
        Time Complexity: O(m * n) - Visit each cell at most once
        Space Complexity: O(m * n) - Worst case recursion stack
        
        Analysis:
        - Pros: Simple to implement, intuitive
        - Cons: Uses recursion stack space
        """
        if not grid:
            return 0
        
        m, n = len(grid), len(grid[0])
        count = 0
        
        def dfs(i, j):
            # Base case: out of bounds or water
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == '0':
                return
            
            # Mark as visited
            grid[i][j] = '0'
            
            # Explore all four directions
            dfs(i + 1, j)  # Down
            dfs(i - 1, j)  # Up
            dfs(i, j + 1)  # Right
            dfs(i, j - 1)  # Left
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1
                    dfs(i, j)
        
        return count
    
    def num_islands_bfs(self, grid: List[List[str]]) -> int:
        """
        Approach 2: Breadth-First Search (BFS) - Iterative
        
        Algorithm:
        1. Use a queue to explore islands level by level
        2. Add all adjacent land cells to queue
        3. Mark visited cells as '0'
        4. Count the number of BFS calls
        
        Time Complexity: O(m * n) - Visit each cell at most once
        Space Complexity: O(min(m, n)) - Queue size
        
        Analysis:
        - Pros: No recursion, better space complexity
        - Cons: Slightly more complex implementation
        """
        if not grid:
            return 0
        
        m, n = len(grid), len(grid[0])
        count = 0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def bfs(i, j):
            queue = deque([(i, j)])
            grid[i][j] = '0'  # Mark as visited
            
            while queue:
                curr_i, curr_j = queue.popleft()
                
                for di, dj in directions:
                    ni, nj = curr_i + di, curr_j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and 
                        grid[ni][nj] == '1'):
                        grid[ni][nj] = '0'
                        queue.append((ni, nj))
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1
                    bfs(i, j)
        
        return count
    
    def num_islands_union_find(self, grid: List[List[str]]) -> int:
        """
        Approach 3: Union Find (Disjoint Set) - Advanced
        
        Algorithm:
        1. Use Union Find data structure
        2. Initially, each '1' is its own set
        3. Union adjacent land cells
        4. Count the number of connected components
        
        Time Complexity: O(m * n * α(m*n)) - α is inverse Ackermann function
        Space Complexity: O(m * n) - Union Find arrays
        
        Analysis:
        - Pros: Efficient for dynamic connectivity
        - Cons: More complex implementation
        """
        if not grid:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        class UnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [0] * size
                self.count = 0
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px != py:
                    if self.rank[px] < self.rank[py]:
                        px, py = py, px
                    self.parent[py] = px
                    if self.rank[px] == self.rank[py]:
                        self.rank[px] += 1
                    self.count -= 1
        
        # Count initial land cells
        land_count = sum(row.count('1') for row in grid)
        uf = UnionFind(m * n)
        uf.count = land_count
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    curr = i * n + j
                    
                    # Check right neighbor
                    if j + 1 < n and grid[i][j + 1] == '1':
                        uf.union(curr, i * n + j + 1)
                    
                    # Check down neighbor
                    if i + 1 < m and grid[i + 1][j] == '1':
                        uf.union(curr, (i + 1) * n + j)
        
        return uf.count
    
    def num_islands_iterative_dfs(self, grid: List[List[str]]) -> int:
        """
        Approach 4: Iterative DFS with Stack
        
        Algorithm:
        1. Use a stack to simulate recursion
        2. Push adjacent cells to stack
        3. Process until stack is empty
        
        Time Complexity: O(m * n) - Visit each cell at most once
        Space Complexity: O(m * n) - Stack size
        
        Analysis:
        - Pros: No recursion, explicit control
        - Cons: Uses stack space
        """
        if not grid:
            return 0
        
        m, n = len(grid), len(grid[0])
        count = 0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def dfs_iterative(i, j):
            stack = [(i, j)]
            grid[i][j] = '0'
            
            while stack:
                curr_i, curr_j = stack.pop()
                
                for di, dj in directions:
                    ni, nj = curr_i + di, curr_j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and 
                        grid[ni][nj] == '1'):
                        grid[ni][nj] = '0'
                        stack.append((ni, nj))
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1
                    dfs_iterative(i, j)
        
        return count
    
    def num_islands_brute_force(self, grid: List[List[str]]) -> int:
        """
        Approach 5: Brute Force - Check each cell independently
        
        Algorithm:
        1. For each '1', check if it's connected to any other '1'
        2. Count isolated '1's
        3. Very inefficient approach
        
        Time Complexity: O(m² * n²) - Check each pair of cells
        Space Complexity: O(1) - No extra space
        
        Analysis:
        - Pros: Simple logic
        - Cons: Extremely inefficient, only for educational purposes
        """
        if not grid:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        def is_connected(i1, j1, i2, j2):
            """Check if two cells are connected (simplified)"""
            if grid[i1][j1] != '1' or grid[i2][j2] != '1':
                return False
            
            # Check if adjacent
            return (abs(i1 - i2) == 1 and j1 == j2) or (abs(j1 - j2) == 1 and i1 == i2)
        
        islands = set()
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    # Find all connected cells
                    connected = set()
                    stack = [(i, j)]
                    
                    while stack:
                        curr_i, curr_j = stack.pop()
                        if (curr_i, curr_j) not in connected:
                            connected.add((curr_i, curr_j))
                            
                            # Check all adjacent cells
                            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                                ni, nj = curr_i + di, curr_j + dj
                                if (0 <= ni < m and 0 <= nj < n and 
                                    grid[ni][nj] == '1'):
                                    stack.append((ni, nj))
                    
                    # Add the island (represented by its top-left cell)
                    islands.add(min(connected))
        
        return len(islands)


# Testing and Benchmarking

def test_number_of_islands():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "grid": [
                ["1","1","1","1","0"],
                ["1","1","0","1","0"],
                ["1","1","0","0","0"],
                ["0","0","0","0","0"]
            ],
            "expected": 1,
            "description": "Single island"
        },
        {
            "grid": [
                ["1","1","0","0","0"],
                ["1","1","0","0","0"],
                ["0","0","1","0","0"],
                ["0","0","0","1","1"]
            ],
            "expected": 3,
            "description": "Three islands"
        },
        {
            "grid": [["1"]],
            "expected": 1,
            "description": "Single cell island"
        },
        {
            "grid": [["0"]],
            "expected": 0,
            "description": "No islands"
        },
        {
            "grid": [
                ["1","0","1","0","1"],
                ["0","1","0","1","0"],
                ["1","0","1","0","1"]
            ],
            "expected": 6,
            "description": "Multiple small islands"
        }
    ]
    
    approaches = [
        ("DFS", solution.num_islands_dfs),
        ("BFS", solution.num_islands_bfs),
        ("Union Find", solution.num_islands_union_find),
        ("Iterative DFS", solution.num_islands_iterative_dfs)
    ]
    
    print("Number of Islands - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Grid size: {len(test['grid'])}x{len(test['grid'][0])}")
        print(f"Expected: {test['expected']} islands")
        
        for approach_name, approach_func in approaches:
            try:
                # Create a copy of the grid for each test
                grid_copy = [row[:] for row in test['grid']]
                result = approach_func(grid_copy)
                
                passed = result == test['expected']
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} islands {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("DFS", "O(m * n)", "O(m * n)", "Simple, uses recursion"),
        ("BFS", "O(m * n)", "O(min(m, n))", "Better space complexity"),
        ("Union Find", "O(m * n * α)", "O(m * n)", "Efficient for dynamic changes"),
        ("Iterative DFS", "O(m * n)", "O(m * n)", "No recursion"),
        ("Brute Force", "O(m² * n²)", "O(1)", "Educational only")
    ]
    
    print(f"{'Approach':<15} {'Time':<15} {'Space':<15} {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<15} {time_comp:<15} {space_comp:<15} {notes}")


if __name__ == "__main__":
    print("Number of Islands - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_number_of_islands()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the DFS approach for optimal performance.")
