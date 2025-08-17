"""
Valid Anagram - LeetCode Problem 242
https://leetcode.com/problems/valid-anagram/

Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, 
typically using all the original letters exactly once.

Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

Example 2:
Input: s = "rat", t = "car"
Output: false

Constraints:
- 1 <= s.length, t.length <= 5 * 10^4
- s and t consist of lowercase English letters.
"""

from typing import List
import time
from collections import Counter, defaultdict


class Solution:
    """
    Solution class with multiple approaches to solve Valid Anagram
    """
    
    def isAnagram(self, s: str, t: str) -> bool:
        """
        Main method - returns the most efficient solution
        """
        return self.valid_anagram_counter(s, t)
    
    def valid_anagram_sorting(self, s: str, t: str) -> bool:
        """
        Approach 1: Sorting - Compare sorted strings
        
        Algorithm:
        1. Sort both strings
        2. Compare the sorted strings
        3. If they are equal, they are anagrams
        
        Time Complexity: O(n log n) - Due to sorting
        Space Complexity: O(n) - Need to store sorted strings
        
        Analysis:
        - Pros: Simple to understand and implement
        - Cons: Slower than hash map approach, requires extra space
        """
        return sorted(s) == sorted(t)
    
    def valid_anagram_counter(self, s: str, t: str) -> bool:
        """
        Approach 2: Counter - Count character frequencies
        
        Algorithm:
        1. Use Counter to count characters in both strings
        2. Compare the counters
        3. If they are equal, strings are anagrams
        
        Time Complexity: O(n) - Single pass through both strings
        Space Complexity: O(1) - Fixed size since we have limited character set
        
        Analysis:
        - Pros: Optimal time complexity, clean implementation
        - Cons: Uses Counter class (but very efficient)
        """
        return Counter(s) == Counter(t)
    
    def valid_anagram_hash_map(self, s: str, t: str) -> bool:
        """
        Approach 3: Hash Map - Manual character counting
        
        Algorithm:
        1. Create hash map to count characters in first string
        2. Decrement counts for characters in second string
        3. Check if all counts are zero
        
        Time Complexity: O(n) - Single pass through both strings
        Space Complexity: O(1) - Fixed size hash map for 26 letters
        
        Analysis:
        - Pros: Shows manual implementation, good for understanding
        - Cons: More verbose than Counter approach
        """
        if len(s) != len(t):
            return False
        
        char_count = defaultdict(int)
        
        # Count characters in first string
        for char in s:
            char_count[char] += 1
        
        # Decrement counts for second string
        for char in t:
            char_count[char] -= 1
            if char_count[char] < 0:
                return False
        
        return True
    
    def valid_anagram_array(self, s: str, t: str) -> bool:
        """
        Approach 4: Array - Use fixed-size array for character counts
        
        Algorithm:
        1. Use array of size 26 to count characters (assuming lowercase letters)
        2. Increment for first string, decrement for second string
        3. Check if all counts are zero
        
        Time Complexity: O(n) - Single pass through both strings
        Space Complexity: O(1) - Fixed size array of 26 elements
        
        Analysis:
        - Pros: Most space efficient, very fast
        - Cons: Only works for lowercase letters, less flexible
        """
        if len(s) != len(t):
            return False
        
        # Array to count characters (assuming lowercase letters)
        char_count = [0] * 26
        
        # Count characters in both strings
        for char in s:
            char_count[ord(char) - ord('a')] += 1
        
        for char in t:
            char_count[ord(char) - ord('a')] -= 1
            if char_count[ord(char) - ord('a')] < 0:
                return False
        
        return True
    
    def valid_anagram_brute_force(self, s: str, t: str) -> bool:
        """
        Approach 5: Brute Force - Check all permutations (for educational purposes)
        
        Note: This is not practical for real use due to factorial complexity
        but shows the concept of what an anagram is.
        
        Time Complexity: O(n!) - All possible permutations
        Space Complexity: O(n) - Recursion stack
        """
        if len(s) != len(t):
            return False
        
        def get_permutations(s):
            if len(s) <= 1:
                return [s]
            
            perms = []
            for i in range(len(s)):
                char = s[i]
                remaining = s[:i] + s[i+1:]
                for perm in get_permutations(remaining):
                    perms.append(char + perm)
            return perms
        
        # This is extremely inefficient and only for demonstration
        # In practice, never use this approach
        return t in get_permutations(s)


# Testing and Benchmarking

def test_valid_anagram():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "s": "anagram",
            "t": "nagaram",
            "expected": True,
            "description": "Valid anagram"
        },
        {
            "s": "rat",
            "t": "car",
            "expected": False,
            "description": "Not an anagram"
        },
        {
            "s": "listen",
            "t": "silent",
            "expected": True,
            "description": "Another valid anagram"
        },
        {
            "s": "hello",
            "t": "world",
            "expected": False,
            "description": "Different lengths"
        },
        {
            "s": "",
            "t": "",
            "expected": True,
            "description": "Empty strings"
        },
        {
            "s": "a",
            "t": "a",
            "expected": True,
            "description": "Single character"
        },
        {
            "s": "a",
            "t": "b",
            "expected": False,
            "description": "Single different characters"
        },
        {
            "s": "abcdef",
            "t": "fedcba",
            "expected": True,
            "description": "Reversed string"
        }
    ]
    
    approaches = [
        ("Sorting", solution.valid_anagram_sorting),
        ("Counter", solution.valid_anagram_counter),
        ("Hash Map", solution.valid_anagram_hash_map),
        ("Array", solution.valid_anagram_array)
    ]
    
    print("Valid Anagram - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: s = '{test['s']}', t = '{test['t']}'")
        print(f"Expected: {test['expected']}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(test['s'], test['t'])
                
                passed = result == test['expected']
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with large input"""
    
    solution = Solution()
    
    # Create large test case
    n = 10000
    s = "a" * n + "b" * n
    t = "b" * n + "a" * n  # Valid anagram
    
    print(f"\nBenchmarking with strings of length {len(s)}")
    print("=" * 50)
    
    approaches = [
        ("Counter", solution.valid_anagram_counter),
        ("Hash Map", solution.valid_anagram_hash_map),
        ("Array", solution.valid_anagram_array),
        ("Sorting", solution.valid_anagram_sorting)
    ]
    
    # Skip brute force for large input as it's too slow
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(s, t)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds (Result: {result})")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Sorting", "O(n log n)", "O(n)", "Simple but slower"),
        ("Counter", "O(n)", "O(1)", "Optimal, clean implementation"),
        ("Hash Map", "O(n)", "O(1)", "Manual implementation"),
        ("Array", "O(n)", "O(1)", "Most space efficient"),
        ("Brute Force", "O(n!)", "O(n)", "Educational only")
    ]
    
    print(f"{'Approach':<15} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 55)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<15} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Valid Anagram - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_valid_anagram()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the Counter approach for optimal performance.")
