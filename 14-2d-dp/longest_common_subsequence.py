"""
Longest Common Subsequence - LeetCode Problem 1143
https://leetcode.com/problems/longest-common-subsequence/

Given two strings text1 and text2, return the length of their longest common subsequence. 
If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some 
characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".

Example 1:
Input: text1 = "abcde", text2 = "ace"
Output: 3
Explanation: The longest common subsequence is "ace" and its length is 3.

Example 2:
Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.

Example 3:
Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.

Constraints:
- 1 <= text1.length, text2.length <= 1000
- text1 and text2 consist of only lowercase English characters.
"""

from typing import List
import time


class Solution:
    """
    Solution class with multiple approaches to solve Longest Common Subsequence
    """
    
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """
        Main method - returns the most efficient solution
        """
        return self.lcs_dp_optimized(text1, text2)
    
    def lcs_dp_optimized(self, text1: str, text2: str) -> int:
        """
        Approach 1: Dynamic Programming with Space Optimization
        
        Algorithm:
        1. Use 2D DP table: dp[i][j] = LCS of text1[0:i] and text2[0:j]
        2. If characters match: dp[i][j] = dp[i-1][j-1] + 1
        3. If characters don't match: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        4. Use only 2 rows to optimize space
        
        Time Complexity: O(m * n) - Fill DP table
        Space Complexity: O(min(m, n)) - Only 2 rows
        
        Analysis:
        - Pros: Optimal time complexity, space optimized
        - Cons: Slightly more complex implementation
        """
        m, n = len(text1), len(text2)
        
        # Use shorter string for columns to minimize space
        if m < n:
            text1, text2 = text2, text1
            m, n = n, m
        
        # Use only 2 rows
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            
            # Swap rows
            prev, curr = curr, prev
        
        return prev[n]
    
    def lcs_dp_standard(self, text1: str, text2: str) -> int:
        """
        Approach 2: Standard Dynamic Programming
        
        Algorithm:
        1. Use 2D DP table: dp[i][j] = LCS of text1[0:i] and text2[0:j]
        2. If characters match: dp[i][j] = dp[i-1][j-1] + 1
        3. If characters don't match: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        Time Complexity: O(m * n) - Fill DP table
        Space Complexity: O(m * n) - Full DP table
        
        Analysis:
        - Pros: Clear logic, easy to understand
        - Cons: Uses more space than necessary
        """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def lcs_memoization(self, text1: str, text2: str) -> int:
        """
        Approach 3: Recursive with Memoization (Top-down DP)
        
        Algorithm:
        1. Use recursion with memoization
        2. Base cases: empty strings
        3. If characters match, recurse on remaining strings
        4. If characters don't match, take max of two possibilities
        
        Time Complexity: O(m * n) - Each subproblem solved once
        Space Complexity: O(m * n) - Memoization cache + recursion stack
        
        Analysis:
        - Pros: Top-down approach, natural recursion
        - Cons: Uses recursion stack space
        """
        memo = {}
        
        def lcs_helper(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i == 0 or j == 0:
                return 0
            
            if text1[i - 1] == text2[j - 1]:
                result = lcs_helper(i - 1, j - 1) + 1
            else:
                result = max(lcs_helper(i - 1, j), lcs_helper(i, j - 1))
            
            memo[(i, j)] = result
            return result
        
        return lcs_helper(len(text1), len(text2))
    
    def lcs_recursive(self, text1: str, text2: str) -> int:
        """
        Approach 4: Pure Recursive (Brute Force)
        
        Algorithm:
        1. Use pure recursion without memoization
        2. Try all possible subsequences
        3. Very inefficient but shows the concept
        
        Time Complexity: O(2^(m+n)) - Exponential
        Space Complexity: O(m + n) - Recursion stack
        
        Analysis:
        - Pros: Simple to understand
        - Cons: Extremely inefficient, only for educational purposes
        """
        def lcs_recursive_helper(i, j):
            if i == 0 or j == 0:
                return 0
            
            if text1[i - 1] == text2[j - 1]:
                return lcs_recursive_helper(i - 1, j - 1) + 1
            else:
                return max(lcs_recursive_helper(i - 1, j), 
                          lcs_recursive_helper(i, j - 1))
        
        return lcs_recursive_helper(len(text1), len(text2))
    
    def lcs_brute_force(self, text1: str, text2: str) -> int:
        """
        Approach 5: Brute Force - Generate all subsequences
        
        Algorithm:
        1. Generate all possible subsequences of both strings
        2. Find the longest common one
        3. Extremely inefficient approach
        
        Time Complexity: O(2^m * 2^n) - Generate all subsequences
        Space Complexity: O(2^m + 2^n) - Store all subsequences
        
        Analysis:
        - Pros: Shows all possibilities
        - Cons: Extremely inefficient, only for educational purposes
        """
        def generate_subsequences(s):
            n = len(s)
            subsequences = []
            
            for i in range(1 << n):  # 2^n possibilities
                subseq = ""
                for j in range(n):
                    if i & (1 << j):
                        subseq += s[j]
                subsequences.append(subseq)
            
            return subsequences
        
        # Generate all subsequences
        subseq1 = set(generate_subsequences(text1))
        subseq2 = set(generate_subsequences(text2))
        
        # Find longest common subsequence
        max_length = 0
        for subseq in subseq1:
            if subseq in subseq2:
                max_length = max(max_length, len(subseq))
        
        return max_length


# Testing and Benchmarking

def test_longest_common_subsequence():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "text1": "abcde",
            "text2": "ace",
            "expected": 3,
            "description": "Basic case"
        },
        {
            "text1": "abc",
            "text2": "abc",
            "expected": 3,
            "description": "Identical strings"
        },
        {
            "text1": "abc",
            "text2": "def",
            "expected": 0,
            "description": "No common subsequence"
        },
        {
            "text1": "abcba",
            "text2": "abcbcba",
            "expected": 5,
            "description": "Longer strings"
        },
        {
            "text1": "a",
            "text2": "a",
            "expected": 1,
            "description": "Single character"
        },
        {
            "text1": "a",
            "text2": "b",
            "expected": 0,
            "description": "Different single characters"
        }
    ]
    
    approaches = [
        ("DP Optimized", solution.lcs_dp_optimized),
        ("DP Standard", solution.lcs_dp_standard),
        ("Memoization", solution.lcs_memoization)
    ]
    
    print("Longest Common Subsequence - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: text1 = '{test['text1']}', text2 = '{test['text2']}'")
        print(f"Expected: {test['expected']}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(test['text1'], test['text2'])
                
                passed = result == test['expected']
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with larger input"""
    
    solution = Solution()
    
    # Test with larger input
    text1 = "abcdefghijklmnopqrstuvwxyz" * 10
    text2 = "zyxwvutsrqponmlkjihgfedcba" * 10
    
    print(f"\nBenchmarking with strings of length {len(text1)} and {len(text2)}")
    print("=" * 50)
    
    approaches = [
        ("DP Optimized", solution.lcs_dp_optimized),
        ("DP Standard", solution.lcs_dp_standard),
        ("Memoization", solution.lcs_memoization)
    ]
    
    # Skip recursive and brute force for large input as they're too slow
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(text1, text2)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds (Result: {result})")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("DP Optimized", "O(m * n)", "O(min(m, n))", "Optimal space complexity"),
        ("DP Standard", "O(m * n)", "O(m * n)", "Clear logic"),
        ("Memoization", "O(m * n)", "O(m * n)", "Top-down approach"),
        ("Recursive", "O(2^(m+n))", "O(m + n)", "Pure recursion"),
        ("Brute Force", "O(2^m * 2^n)", "O(2^m + 2^n)", "Educational only")
    ]
    
    print(f"{'Approach':<15} {'Time':<15} {'Space':<15} {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<15} {time_comp:<15} {space_comp:<15} {notes}")


if __name__ == "__main__":
    print("Longest Common Subsequence - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_longest_common_subsequence()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the DP optimized approach for optimal performance.")
