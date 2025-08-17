"""
Course Schedule - LeetCode Problem 207
https://leetcode.com/problems/course-schedule/

There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. 
You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you 
must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

Return true if you can finish all courses. Otherwise, return false.

Example 1:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.

Example 2:
Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also 
have finished course 1. So it is impossible.

Constraints:
- 1 <= numCourses <= 2000
- 0 <= prerequisites.length <= 5000
- prerequisites[i].length == 2
- 0 <= ai, bi < numCourses
- All the pairs prerequisites[i] are unique.
"""

from typing import List
import time
from collections import defaultdict, deque


class Solution:
    """
    Solution class with multiple approaches to solve Course Schedule
    """
    
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Main method - returns the most efficient solution
        """
        return self.can_finish_dfs(numCourses, prerequisites)
    
    def can_finish_dfs(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Approach 1: DFS with Cycle Detection
        
        Algorithm:
        1. Build adjacency list from prerequisites
        2. Use DFS to detect cycles
        3. If cycle exists, return False (impossible to complete)
        4. Use visited and recursion stack to track cycles
        
        Time Complexity: O(V + E) - V vertices, E edges
        Space Complexity: O(V + E) - Adjacency list + recursion stack
        
        Analysis:
        - Pros: Intuitive approach, detects cycles efficiently
        - Cons: Uses recursion stack space
        """
        # Build adjacency list
        graph = defaultdict(list)
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        # Track visited nodes and recursion stack
        visited = [False] * numCourses
        recursion_stack = [False] * numCourses
        
        def has_cycle(node):
            visited[node] = True
            recursion_stack[node] = True
            
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    if has_cycle(neighbor):
                        return True
                elif recursion_stack[neighbor]:
                    return True
            
            recursion_stack[node] = False
            return False
        
        # Check for cycles starting from each unvisited node
        for course in range(numCourses):
            if not visited[course]:
                if has_cycle(course):
                    return False
        
        return True
    
    def can_finish_bfs(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Approach 2: BFS with Topological Sort (Kahn's Algorithm)
        
        Algorithm:
        1. Build adjacency list and in-degree count
        2. Use BFS to process nodes with in-degree 0
        3. Reduce in-degree of neighbors
        4. If all nodes processed, return True
        
        Time Complexity: O(V + E) - V vertices, E edges
        Space Complexity: O(V + E) - Adjacency list + queue
        
        Analysis:
        - Pros: No recursion, clear topological sort
        - Cons: Slightly more complex implementation
        """
        # Build adjacency list and in-degree count
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # BFS with queue
        queue = deque()
        for course in range(numCourses):
            if in_degree[course] == 0:
                queue.append(course)
        
        courses_completed = 0
        
        while queue:
            course = queue.popleft()
            courses_completed += 1
            
            for neighbor in graph[course]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return courses_completed == numCourses
    
    def can_finish_coloring(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Approach 3: DFS with Coloring (3-Color Algorithm)
        
        Algorithm:
        1. Use 3 colors: WHITE (unvisited), GRAY (in recursion), BLACK (completed)
        2. DFS with coloring to detect cycles
        3. If GRAY node is encountered, cycle exists
        
        Time Complexity: O(V + E) - V vertices, E edges
        Space Complexity: O(V + E) - Adjacency list + color array
        
        Analysis:
        - Pros: Clear cycle detection, efficient
        - Cons: More complex color management
        """
        # Color constants
        WHITE, GRAY, BLACK = 0, 1, 2
        
        # Build adjacency list
        graph = defaultdict(list)
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        # Color array
        colors = [WHITE] * numCourses
        
        def has_cycle(node):
            if colors[node] == GRAY:
                return True  # Back edge found
            if colors[node] == BLACK:
                return False  # Already processed
            
            colors[node] = GRAY
            
            for neighbor in graph[node]:
                if has_cycle(neighbor):
                    return True
            
            colors[node] = BLACK
            return False
        
        # Check for cycles starting from each unvisited node
        for course in range(numCourses):
            if colors[course] == WHITE:
                if has_cycle(course):
                    return False
        
        return True
    
    def can_finish_union_find(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Approach 4: Union Find (Not suitable for this problem)
        
        Algorithm:
        1. Use Union Find to detect cycles
        2. This approach is not suitable for directed graphs
        3. Included for educational purposes
        
        Time Complexity: O(E * α(V)) - E edges, α is inverse Ackermann
        Space Complexity: O(V) - Union Find arrays
        
        Analysis:
        - Pros: Efficient for undirected graphs
        - Cons: Not suitable for directed graphs, included for education
        """
        class UnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [0] * size
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return False  # Cycle detected
                
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
                return True
        
        # Note: This approach doesn't work correctly for directed graphs
        # It's included only for educational purposes
        uf = UnionFind(numCourses)
        
        for course, prereq in prerequisites:
            if not uf.union(course, prereq):
                return False
        
        return True
    
    def can_finish_brute_force(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Approach 5: Brute Force - Check all possible orderings
        
        Algorithm:
        1. Generate all possible course orderings
        2. Check if any ordering satisfies all prerequisites
        3. Extremely inefficient approach
        
        Time Complexity: O(n!) - Generate all permutations
        Space Complexity: O(n!) - Store all permutations
        
        Analysis:
        - Pros: Guaranteed to find solution if exists
        - Cons: Extremely inefficient, only for educational purposes
        """
        from itertools import permutations
        
        # Build prerequisite map
        prereq_map = defaultdict(set)
        for course, prereq in prerequisites:
            prereq_map[course].add(prereq)
        
        # Check all possible orderings
        for ordering in permutations(range(numCourses)):
            valid = True
            for course in ordering:
                for prereq in prereq_map[course]:
                    if prereq not in ordering[:ordering.index(course)]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return True
        
        return False


# Testing and Benchmarking

def test_course_schedule():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "numCourses": 2,
            "prerequisites": [[1, 0]],
            "expected": True,
            "description": "Simple valid case"
        },
        {
            "numCourses": 2,
            "prerequisites": [[1, 0], [0, 1]],
            "expected": False,
            "description": "Cycle detected"
        },
        {
            "numCourses": 4,
            "prerequisites": [[1, 0], [2, 1], [3, 2]],
            "expected": True,
            "description": "Linear dependency"
        },
        {
            "numCourses": 3,
            "prerequisites": [[1, 0], [2, 1], [0, 2]],
            "expected": False,
            "description": "Circular dependency"
        },
        {
            "numCourses": 1,
            "prerequisites": [],
            "expected": True,
            "description": "Single course"
        },
        {
            "numCourses": 3,
            "prerequisites": [[1, 0], [2, 0]],
            "expected": True,
            "description": "Multiple courses from same prereq"
        }
    ]
    
    approaches = [
        ("DFS", solution.can_finish_dfs),
        ("BFS", solution.can_finish_bfs),
        ("Coloring", solution.can_finish_coloring)
    ]
    
    print("Course Schedule - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: numCourses = {test['numCourses']}, prerequisites = {test['prerequisites']}")
        print(f"Expected: {test['expected']}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(test['numCourses'], test['prerequisites'])
                
                passed = result == test['expected']
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with larger input"""
    
    solution = Solution()
    
    # Test with larger input
    numCourses = 100
    prerequisites = []
    
    # Create a valid course schedule
    for i in range(numCourses - 1):
        prerequisites.append([i + 1, i])
    
    print(f"\nBenchmarking with {numCourses} courses and {len(prerequisites)} prerequisites")
    print("=" * 50)
    
    approaches = [
        ("DFS", solution.can_finish_dfs),
        ("BFS", solution.can_finish_bfs),
        ("Coloring", solution.can_finish_coloring)
    ]
    
    # Skip brute force for large input as it's too slow
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(numCourses, prerequisites)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds (Result: {result})")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("DFS", "O(V + E)", "O(V + E)", "Intuitive cycle detection"),
        ("BFS", "O(V + E)", "O(V + E)", "Topological sort approach"),
        ("Coloring", "O(V + E)", "O(V + E)", "3-color cycle detection"),
        ("Union Find", "O(E * α(V))", "O(V)", "Not suitable for directed graphs"),
        ("Brute Force", "O(V!)", "O(V!)", "Educational only")
    ]
    
    print(f"{'Approach':<15} {'Time':<15} {'Space':<15} {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<15} {time_comp:<15} {space_comp:<15} {notes}")


if __name__ == "__main__":
    print("Course Schedule - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_course_schedule()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the DFS approach for optimal performance.")
