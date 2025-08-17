"""
Binary Tree Inorder Traversal - LeetCode Problem 94
https://leetcode.com/problems/binary-tree-inorder-traversal/

Given the root of a binary tree, return the inorder traversal of its nodes' values.

Example 1:
Input: root = [1,null,2,3]
Output: [1,3,2]

Example 2:
Input: root = []
Output: []

Example 3:
Input: root = [1]
Output: [1]

Constraints:
- The number of nodes in the tree is in the range [0, 100].
- -100 <= Node.val <= 100
"""

from typing import List, Optional
import time


class TreeNode:
    """Definition for a binary tree node"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    """
    Solution class with multiple approaches to solve Binary Tree Inorder Traversal
    """
    
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        Main method - returns the most efficient solution
        """
        return self.inorder_traversal_iterative(root)
    
    def inorder_traversal_recursive(self, root: Optional[TreeNode]) -> List[int]:
        """
        Approach 1: Recursive - Simple and intuitive
        
        Algorithm:
        1. Recursively traverse left subtree
        2. Visit current node (add to result)
        3. Recursively traverse right subtree
        
        Time Complexity: O(n) - Visit each node exactly once
        Space Complexity: O(h) - Height of the tree (recursion stack)
        
        Analysis:
        - Pros: Simple to understand and implement
        - Cons: Uses recursion stack space
        """
        result = []
        
        def inorder_helper(node):
            if not node:
                return
            
            inorder_helper(node.left)
            result.append(node.val)
            inorder_helper(node.right)
        
        inorder_helper(root)
        return result
    
    def inorder_traversal_iterative(self, root: Optional[TreeNode]) -> List[int]:
        """
        Approach 2: Iterative with Stack - Most efficient
        
        Algorithm:
        1. Use a stack to simulate recursion
        2. Push all left children to stack
        3. Pop and visit node, then push right child
        4. Repeat until stack is empty
        
        Time Complexity: O(n) - Visit each node exactly once
        Space Complexity: O(h) - Height of the tree (stack space)
        
        Analysis:
        - Pros: No recursion, more control over space usage
        - Cons: Slightly more complex implementation
        """
        result = []
        stack = []
        current = root
        
        while current or stack:
            # Push all left children to stack
            while current:
                stack.append(current)
                current = current.left
            
            # Pop and visit node
            current = stack.pop()
            result.append(current.val)
            
            # Move to right child
            current = current.right
        
        return result
    
    def inorder_traversal_morris(self, root: Optional[TreeNode]) -> List[int]:
        """
        Approach 3: Morris Traversal - Constant space
        
        Algorithm:
        1. Use threaded binary tree concept
        2. Create temporary links to predecessor
        3. Traverse without using stack or recursion
        
        Time Complexity: O(n) - Visit each node at most twice
        Space Complexity: O(1) - Only using a few pointers
        
        Analysis:
        - Pros: Constant space complexity
        - Cons: Modifies tree structure temporarily, complex
        """
        result = []
        current = root
        
        while current:
            if not current.left:
                # No left child, visit current node
                result.append(current.val)
                current = current.right
            else:
                # Find inorder predecessor
                predecessor = current.left
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    # Create temporary link
                    predecessor.right = current
                    current = current.left
                else:
                    # Remove temporary link and visit current
                    predecessor.right = None
                    result.append(current.val)
                    current = current.right
        
        return result
    
    def inorder_traversal_dfs(self, root: Optional[TreeNode]) -> List[int]:
        """
        Approach 4: DFS with explicit stack - Alternative iterative
        
        Algorithm:
        1. Use depth-first search with explicit stack
        2. Track visited nodes to avoid infinite loops
        3. Process nodes in inorder sequence
        
        Time Complexity: O(n) - Visit each node exactly once
        Space Complexity: O(n) - Stack and visited set
        
        Analysis:
        - Pros: Clear DFS approach, easy to understand
        - Cons: Uses extra space for visited tracking
        """
        if not root:
            return []
        
        result = []
        stack = [(root, False)]  # (node, visited)
        
        while stack:
            node, visited = stack.pop()
            
            if visited:
                result.append(node.val)
            else:
                # Push in reverse order: right, current, left
                if node.right:
                    stack.append((node.right, False))
                stack.append((node, True))
                if node.left:
                    stack.append((node.left, False))
        
        return result
    
    def inorder_traversal_brute_force(self, root: Optional[TreeNode]) -> List[int]:
        """
        Approach 5: Brute Force - Collect all nodes and sort (for educational purposes)
        
        Algorithm:
        1. Collect all nodes in any order
        2. Sort by their values
        3. Return sorted values
        
        Time Complexity: O(n log n) - Due to sorting
        Space Complexity: O(n) - Store all nodes
        
        Analysis:
        - Pros: Simple to implement
        - Cons: Incorrect for tree structure, inefficient
        """
        if not root:
            return []
        
        def collect_nodes(node):
            if not node:
                return []
            return [node.val] + collect_nodes(node.left) + collect_nodes(node.right)
        
        nodes = collect_nodes(root)
        return sorted(nodes)  # This is incorrect for inorder traversal!


# Helper functions for testing

def create_binary_tree(values: list) -> Optional[TreeNode]:
    """Create a binary tree from a list of values (level-order)"""
    if not values:
        return None
    
    root = TreeNode(values[0])
    queue = [root]
    i = 1
    
    while queue and i < len(values):
        node = queue.pop(0)
        
        # Left child
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        
        # Right child
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    
    return root


def tree_to_list(root: Optional[TreeNode]) -> list:
    """Convert binary tree to list (level-order) for testing"""
    if not root:
        return []
    
    result = []
    queue = [root]
    
    while queue:
        node = queue.pop(0)
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)
    
    # Remove trailing None values
    while result and result[-1] is None:
        result.pop()
    
    return result


# Testing and Benchmarking

def test_inorder_traversal():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "values": [1, None, 2, 3],
            "expected": [1, 3, 2],
            "description": "Basic tree"
        },
        {
            "values": [],
            "expected": [],
            "description": "Empty tree"
        },
        {
            "values": [1],
            "expected": [1],
            "description": "Single node"
        },
        {
            "values": [1, 2, 3, 4, 5],
            "expected": [4, 2, 5, 1, 3],
            "description": "Complete binary tree"
        },
        {
            "values": [1, None, 2, None, 3],
            "expected": [1, 2, 3],
            "description": "Right-skewed tree"
        },
        {
            "values": [3, 1, 4, None, 2],
            "expected": [1, 2, 3, 4],
            "description": "BST-like tree"
        }
    ]
    
    approaches = [
        ("Recursive", solution.inorder_traversal_recursive),
        ("Iterative", solution.inorder_traversal_iterative),
        ("Morris", solution.inorder_traversal_morris),
        ("DFS", solution.inorder_traversal_dfs)
    ]
    
    print("Binary Tree Inorder Traversal - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: values = {test['values']}")
        print(f"Expected: {test['expected']}")
        
        # Create binary tree for this test
        root = create_binary_tree(test['values'])
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(root)
                
                passed = result == test['expected']
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with large input"""
    
    solution = Solution()
    
    # Create large test case (complete binary tree)
    n = 1000
    values = list(range(1, n + 1))
    
    root = create_binary_tree(values)
    
    print(f"\nBenchmarking with binary tree of size {n}")
    print("=" * 50)
    
    approaches = [
        ("Recursive", solution.inorder_traversal_recursive),
        ("Iterative", solution.inorder_traversal_iterative),
        ("Morris", solution.inorder_traversal_morris),
        ("DFS", solution.inorder_traversal_dfs)
    ]
    
    # Skip brute force as it's incorrect
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(root)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds")
        print(f"  Result length: {len(result)}")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Recursive", "O(n)", "O(h)", "Simple, uses recursion stack"),
        ("Iterative", "O(n)", "O(h)", "No recursion, most practical"),
        ("Morris", "O(n)", "O(1)", "Constant space, modifies tree"),
        ("DFS", "O(n)", "O(n)", "Clear approach, extra space"),
        ("Brute Force", "O(n log n)", "O(n)", "Incorrect, educational only")
    ]
    
    print(f"{'Approach':<15} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 55)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<15} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Binary Tree Inorder Traversal - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_inorder_traversal()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the iterative approach for optimal performance.")
