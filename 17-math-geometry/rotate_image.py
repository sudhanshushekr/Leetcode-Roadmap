"""
Rotate Image - LeetCode Problem 48
https://leetcode.com/problems/rotate-image/

You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. 
DO NOT allocate another 2D matrix and do the rotation.

Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

Example 2:
Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

Constraints:
- n == matrix.length == matrix[i].length
- 1 <= n <= 20
- -1000 <= matrix[i][j] <= 1000
"""

from typing import List
import time


class Solution:
    """
    Solution class with multiple approaches to solve Rotate Image
    """
    
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Main method - returns the most efficient solution
        """
        self.rotate_transpose_reverse(matrix)
    
    def rotate_transpose_reverse(self, matrix: List[List[int]]) -> None:
        """
        Approach 1: Transpose + Reverse - Most Efficient
        
        Algorithm:
        1. Transpose the matrix (swap rows and columns)
        2. Reverse each row to get 90-degree clockwise rotation
        
        Time Complexity: O(n²) - Visit each cell twice
        Space Complexity: O(1) - In-place operation
        
        Analysis:
        - Pros: Simple to understand, optimal space complexity
        - Cons: Requires understanding of matrix operations
        """
        n = len(matrix)
        
        # Transpose the matrix
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # Reverse each row
        for i in range(n):
            matrix[i].reverse()
    
    def rotate_layer_by_layer(self, matrix: List[List[int]]) -> None:
        """
        Approach 2: Layer by Layer Rotation
        
        Algorithm:
        1. Rotate the matrix layer by layer from outside to inside
        2. For each layer, rotate 4 elements at a time
        3. Use a temporary variable to store one element
        
        Time Complexity: O(n²) - Visit each cell once
        Space Complexity: O(1) - Only one temporary variable
        
        Analysis:
        - Pros: Intuitive approach, minimal extra space
        - Cons: More complex implementation
        """
        n = len(matrix)
        
        for layer in range(n // 2):
            first = layer
            last = n - 1 - layer
            
            for i in range(first, last):
                offset = i - first
                
                # Save top element
                top = matrix[first][i]
                
                # Move left to top
                matrix[first][i] = matrix[last - offset][first]
                
                # Move bottom to left
                matrix[last - offset][first] = matrix[last][last - offset]
                
                # Move right to bottom
                matrix[last][last - offset] = matrix[i][last]
                
                # Move top to right
                matrix[i][last] = top
    
    def rotate_using_auxiliary_matrix(self, matrix: List[List[int]]) -> None:
        """
        Approach 3: Using Auxiliary Matrix (Not in-place)
        
        Algorithm:
        1. Create a new matrix to store rotated result
        2. Copy elements in rotated positions
        3. Copy back to original matrix
        
        Time Complexity: O(n²) - Visit each cell twice
        Space Complexity: O(n²) - Extra matrix
        
        Analysis:
        - Pros: Simple to implement and understand
        - Cons: Uses extra space (violates problem constraint)
        """
        n = len(matrix)
        rotated = [[0] * n for _ in range(n)]
        
        # Copy elements in rotated positions
        for i in range(n):
            for j in range(n):
                rotated[j][n - 1 - i] = matrix[i][j]
        
        # Copy back to original matrix
        for i in range(n):
            for j in range(n):
                matrix[i][j] = rotated[i][j]
    
    def rotate_using_rings(self, matrix: List[List[int]]) -> None:
        """
        Approach 4: Ring by Ring Rotation
        
        Algorithm:
        1. Treat the matrix as concentric rings
        2. Rotate each ring independently
        3. Use mathematical formula for rotation
        
        Time Complexity: O(n²) - Visit each cell once
        Space Complexity: O(1) - In-place operation
        
        Analysis:
        - Pros: Mathematical approach, efficient
        - Cons: Complex mathematical understanding required
        """
        n = len(matrix)
        
        for i in range(n // 2):
            for j in range(i, n - 1 - i):
                # Store current element
                temp = matrix[i][j]
                
                # Move elements in 90-degree rotation
                matrix[i][j] = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = temp
    
    def rotate_brute_force(self, matrix: List[List[int]]) -> None:
        """
        Approach 5: Brute Force - Element by Element
        
        Algorithm:
        1. For each element, calculate its new position
        2. Use mathematical formula: (i, j) -> (j, n-1-i)
        3. Swap elements one by one
        
        Time Complexity: O(n²) - Visit each cell
        Space Complexity: O(1) - In-place operation
        
        Analysis:
        - Pros: Direct mathematical approach
        - Cons: More swaps than necessary
        """
        n = len(matrix)
        
        for i in range(n // 2):
            for j in range(i, n - 1 - i):
                # Calculate new positions
                new_i, new_j = j, n - 1 - i
                
                # Swap elements
                matrix[i][j], matrix[new_i][new_j] = matrix[new_i][new_j], matrix[i][j]


# Helper functions for testing

def print_matrix(matrix: List[List[int]]) -> None:
    """Print matrix in a readable format"""
    for row in matrix:
        print(row)


def create_matrix(n: int) -> List[List[int]]:
    """Create an n x n matrix with sequential numbers"""
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(i * n + j + 1)
        matrix.append(row)
    return matrix


# Testing and Benchmarking

def test_rotate_image():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "expected": [[7, 4, 1], [8, 5, 2], [9, 6, 3]],
            "description": "3x3 matrix"
        },
        {
            "matrix": [[1, 2], [3, 4]],
            "expected": [[3, 1], [4, 2]],
            "description": "2x2 matrix"
        },
        {
            "matrix": [[1]],
            "expected": [[1]],
            "description": "1x1 matrix"
        },
        {
            "matrix": [[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]],
            "expected": [[15, 13, 2, 5], [14, 3, 4, 1], [12, 6, 8, 9], [16, 7, 10, 11]],
            "description": "4x4 matrix"
        }
    ]
    
    approaches = [
        ("Transpose + Reverse", solution.rotate_transpose_reverse),
        ("Layer by Layer", solution.rotate_layer_by_layer),
        ("Auxiliary Matrix", solution.rotate_using_auxiliary_matrix),
        ("Ring by Ring", solution.rotate_using_rings),
        ("Brute Force", solution.rotate_brute_force)
    ]
    
    print("Rotate Image - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input matrix:")
        print_matrix(test['matrix'])
        print(f"Expected:")
        print_matrix(test['expected'])
        
        for approach_name, approach_func in approaches:
            try:
                # Create a copy of the matrix for each test
                matrix_copy = [row[:] for row in test['matrix']]
                approach_func(matrix_copy)
                
                # Check if result matches expected
                passed = matrix_copy == test['expected']
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {status}")
                if not passed:
                    print(f"    Result:")
                    print_matrix(matrix_copy)
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with larger input"""
    
    solution = Solution()
    
    # Test with larger matrix
    n = 100
    matrix = create_matrix(n)
    
    print(f"\nBenchmarking with {n}x{n} matrix")
    print("=" * 50)
    
    approaches = [
        ("Transpose + Reverse", solution.rotate_transpose_reverse),
        ("Layer by Layer", solution.rotate_layer_by_layer),
        ("Ring by Ring", solution.rotate_using_rings),
        ("Brute Force", solution.rotate_brute_force)
    ]
    
    # Skip auxiliary matrix for large input as it uses too much space
    for approach_name, approach_func in approaches:
        start_time = time.time()
        matrix_copy = [row[:] for row in matrix]
        approach_func(matrix_copy)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Transpose + Reverse", "O(n²)", "O(1)", "Simple and efficient"),
        ("Layer by Layer", "O(n²)", "O(1)", "Intuitive approach"),
        ("Auxiliary Matrix", "O(n²)", "O(n²)", "Simple but uses extra space"),
        ("Ring by Ring", "O(n²)", "O(1)", "Mathematical approach"),
        ("Brute Force", "O(n²)", "O(1)", "Direct approach")
    ]
    
    print(f"{'Approach':<20} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<20} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Rotate Image - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_rotate_image()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the transpose + reverse approach for optimal performance.")
