"""
Group Anagrams - LeetCode Problem 49
https://leetcode.com/problems/group-anagrams/

Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, 
typically using all the original letters exactly once.

Example 1:
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Example 2:
Input: strs = [""]
Output: [[""]]

Example 3:
Input: strs = ["a"]
Output: [["a"]]

Constraints:
- 1 <= strs.length <= 10^4
- 0 <= strs[i].length <= 100
- strs[i] consists of lowercase English letters.
"""

from typing import List
import time
from collections import defaultdict, Counter


class Solution:
    """
    Solution class with multiple approaches to solve Group Anagrams
    """
    
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        Main method - returns the most efficient solution
        """
        return self.group_anagrams_sorted_key(strs)
    
    def group_anagrams_sorted_key(self, strs: List[str]) -> List[List[str]]:
        """
        Approach 1: Sorted String as Key
        
        Algorithm:
        1. Use sorted string as key for grouping
        2. Group strings with same sorted key together
        3. Return all groups
        
        Time Complexity: O(n * k * log k) - n strings, k is max string length
        Space Complexity: O(n * k) - Store all strings and sorted keys
        
        Analysis:
        - Pros: Simple to understand and implement
        - Cons: Sorting each string is expensive for long strings
        """
        groups = defaultdict(list)
        
        for s in strs:
            # Sort characters to create key
            key = ''.join(sorted(s))
            groups[key].append(s)
        
        return list(groups.values())
    
    def group_anagrams_character_count(self, strs: List[str]) -> List[List[str]]:
        """
        Approach 2: Character Count as Key
        
        Algorithm:
        1. Count characters in each string
        2. Use character count as key (e.g., "a1b2c3")
        3. Group strings with same character count
        
        Time Complexity: O(n * k) - n strings, k is max string length
        Space Complexity: O(n * k) - Store all strings and count keys
        
        Analysis:
        - Pros: No sorting required, more efficient for long strings
        - Cons: More complex key generation
        """
        groups = defaultdict(list)
        
        for s in strs:
            # Count characters
            count = [0] * 26  # Assuming lowercase letters
            for char in s:
                count[ord(char) - ord('a')] += 1
            
            # Create key from count array
            key = '#'.join(map(str, count))
            groups[key].append(s)
        
        return list(groups.values())
    
    def group_anagrams_counter_key(self, strs: List[str]) -> List[List[str]]:
        """
        Approach 3: Counter as Key (Tuple)
        
        Algorithm:
        1. Use Counter to count characters
        2. Convert Counter to tuple for hashing
        3. Group by Counter tuple
        
        Time Complexity: O(n * k) - n strings, k is max string length
        Space Complexity: O(n * k) - Store all strings and Counter tuples
        
        Analysis:
        - Pros: Clean implementation using Counter
        - Cons: Tuple conversion overhead
        """
        groups = defaultdict(list)
        
        for s in strs:
            # Use Counter and convert to tuple for hashing
            count = Counter(s)
            key = tuple(sorted(count.items()))  # Sort for consistent key
            groups[key].append(s)
        
        return list(groups.values())
    
    def group_anagrams_prime_product(self, strs: List[str]) -> List[List[str]]:
        """
        Approach 4: Prime Number Product (Advanced)
        
        Algorithm:
        1. Assign prime numbers to each letter (a=2, b=3, c=5, etc.)
        2. Calculate product of primes for each string
        3. Group by product (anagrams will have same product)
        
        Time Complexity: O(n * k) - n strings, k is max string length
        Space Complexity: O(n) - Store products and strings
        
        Analysis:
        - Pros: Very fast for long strings, unique product for each anagram group
        - Cons: Risk of integer overflow for very long strings
        """
        # Prime numbers for each letter
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
        
        groups = defaultdict(list)
        
        for s in strs:
            # Calculate product of primes
            product = 1
            for char in s:
                product *= primes[ord(char) - ord('a')]
            
            groups[product].append(s)
        
        return list(groups.values())
    
    def group_anagrams_brute_force(self, strs: List[str]) -> List[List[str]]:
        """
        Approach 5: Brute Force - Compare each pair (for educational purposes)
        
        Note: This is extremely inefficient and only for demonstration.
        
        Time Complexity: O(n² * k * log k) - Compare each pair and sort
        Space Complexity: O(n * k) - Store all strings
        """
        if not strs:
            return []
        
        groups = []
        used = [False] * len(strs)
        
        for i in range(len(strs)):
            if used[i]:
                continue
            
            # Start new group
            current_group = [strs[i]]
            used[i] = True
            
            # Find all anagrams of current string
            for j in range(i + 1, len(strs)):
                if not used[j] and self.is_anagram(strs[i], strs[j]):
                    current_group.append(strs[j])
                    used[j] = True
            
            groups.append(current_group)
        
        return groups
    
    def is_anagram(self, s: str, t: str) -> bool:
        """Helper method to check if two strings are anagrams"""
        return sorted(s) == sorted(t)


# Testing and Benchmarking

def test_group_anagrams():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "strs": ["eat", "tea", "tan", "ate", "nat", "bat"],
            "expected": [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]],
            "description": "Multiple anagram groups"
        },
        {
            "strs": [""],
            "expected": [[""]],
            "description": "Single empty string"
        },
        {
            "strs": ["a"],
            "expected": [["a"]],
            "description": "Single character"
        },
        {
            "strs": ["abc", "cba", "bac", "cab", "acb", "bca"],
            "expected": [["abc", "cba", "bac", "cab", "acb", "bca"]],
            "description": "All anagrams of each other"
        },
        {
            "strs": ["abc", "def", "ghi"],
            "expected": [["abc"], ["def"], ["ghi"]],
            "description": "No anagrams"
        },
        {
            "strs": [],
            "expected": [],
            "description": "Empty array"
        }
    ]
    
    approaches = [
        ("Sorted Key", solution.group_anagrams_sorted_key),
        ("Character Count", solution.group_anagrams_character_count),
        ("Counter Key", solution.group_anagrams_counter_key),
        ("Prime Product", solution.group_anagrams_prime_product)
    ]
    
    print("Group Anagrams - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: strs = {test['strs']}")
        print(f"Expected: {test['expected']}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(test['strs'])
                
                # Sort both result and expected for comparison
                result_sorted = [sorted(group) for group in sorted(result, key=lambda x: (len(x), x[0] if x else ''))]
                expected_sorted = [sorted(group) for group in sorted(test['expected'], key=lambda x: (len(x), x[0] if x else ''))]
                
                passed = result_sorted == expected_sorted
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def benchmark_approaches():
    """Benchmark different approaches with large input"""
    
    solution = Solution()
    
    # Create large test case
    n = 1000
    strs = []
    for i in range(n):
        # Create strings with same characters in different order
        base = "abcdefghijklmnopqrstuvwxyz"
        if i % 3 == 0:
            strs.append(base)
        elif i % 3 == 1:
            strs.append(base[::-1])  # Reverse
        else:
            strs.append(base[::2] + base[1::2])  # Interleaved
    
    print(f"\nBenchmarking with {len(strs)} strings")
    print("=" * 50)
    
    approaches = [
        ("Sorted Key", solution.group_anagrams_sorted_key),
        ("Character Count", solution.group_anagrams_character_count),
        ("Counter Key", solution.group_anagrams_counter_key),
        ("Prime Product", solution.group_anagrams_prime_product)
    ]
    
    # Skip brute force for large input as it's too slow
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(strs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds")
        print(f"  Groups found: {len(result)}")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Sorted Key", "O(n * k * log k)", "O(n * k)", "Simple, good for short strings"),
        ("Character Count", "O(n * k)", "O(n * k)", "Efficient for long strings"),
        ("Counter Key", "O(n * k)", "O(n * k)", "Clean implementation"),
        ("Prime Product", "O(n * k)", "O(n)", "Fastest, but risk of overflow"),
        ("Brute Force", "O(n² * k * log k)", "O(n * k)", "Educational only")
    ]
    
    print(f"{'Approach':<20} {'Time':<15} {'Space':<10} {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<20} {time_comp:<15} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Group Anagrams - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_group_anagrams()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use character count approach for optimal performance.")
