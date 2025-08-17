"""
Stack Problems - NeetCode Style Template

This template provides a structured approach to solving stack problems
with multiple solutions, complexity analysis, and clear explanations.

Common Patterns:
1. Monotonic Stack
2. Stack with Hash Map
3. Two Stacks
4. Stack with Queue
5. Parentheses Matching
6. Expression Evaluation
"""

from typing import List, Dict, Optional
from collections import deque
import time


class Solution:
    """
    Main solution class following NeetCode's structure
    """
    
    def problem_name(self, nums: List[int]) -> List[int]:
        """
        Problem: [Brief description of the problem]
        
        Example:
        Input: nums = [1, 2, 3, 4, 5]
        Output: [5, 4, 3, 2, 1]
        
        Approach 1: Brute Force
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        
        # Approach 1: Brute Force Solution
        def brute_force_solution(nums: List[int]) -> List[int]:
            """
            Brute force approach - usually the first solution that comes to mind
            """
            result = []
            for i in range(len(nums)):
                # Your logic here
                pass
            return result
        
        # Approach 2: Stack Solution
        def stack_solution(nums: List[int]) -> List[int]:
            """
            Stack approach - efficient solution
            """
            stack = []
            result = []
            
            for num in nums:
                # Your stack logic here
                pass
            
            return result
        
        # Approach 3: Most Efficient Solution
        def most_efficient_solution(nums: List[int]) -> List[int]:
            """
            Most efficient approach - often involves mathematical insights
            """
            # Mathematical approach or advanced algorithm
            return result
        
        # Return the most efficient solution
        return stack_solution(nums)


# Common Stack Patterns

class StackPatterns:
    """Common patterns and techniques for stack problems"""
    
    @staticmethod
    def next_greater_element(nums: List[int]) -> List[int]:
        """
        Next Greater Element - Monotonic stack
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                result[stack.pop()] = nums[i]
            stack.append(i)
        
        return result
    
    @staticmethod
    def valid_parentheses(s: str) -> bool:
        """
        Valid Parentheses - Stack matching
        Time: O(n), Space: O(n)
        """
        stack = []
        brackets = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in '({[':
                stack.append(char)
            elif char in ')}]':
                if not stack or stack.pop() != brackets[char]:
                    return False
        
        return len(stack) == 0
    
    @staticmethod
    def min_stack_operations() -> 'MinStack':
        """
        Min Stack - Two stacks approach
        Time: O(1) for all operations, Space: O(n)
        """
        class MinStack:
            def __init__(self):
                self.stack = []
                self.min_stack = []
            
            def push(self, val: int) -> None:
                self.stack.append(val)
                if not self.min_stack or val <= self.min_stack[-1]:
                    self.min_stack.append(val)
            
            def pop(self) -> None:
                if self.stack:
                    if self.stack[-1] == self.min_stack[-1]:
                        self.min_stack.pop()
                    self.stack.pop()
            
            def top(self) -> int:
                return self.stack[-1] if self.stack else None
            
            def getMin(self) -> int:
                return self.min_stack[-1] if self.min_stack else None
        
        return MinStack()
    
    @staticmethod
    def evaluate_reverse_polish_notation(tokens: List[str]) -> int:
        """
        Evaluate Reverse Polish Notation
        Time: O(n), Space: O(n)
        """
        stack = []
        operators = {'+', '-', '*', '/'}
        
        for token in tokens:
            if token in operators:
                b = stack.pop()
                a = stack.pop()
                
                if token == '+':
                    stack.append(a + b)
                elif token == '-':
                    stack.append(a - b)
                elif token == '*':
                    stack.append(a * b)
                elif token == '/':
                    stack.append(int(a / b))
            else:
                stack.append(int(token))
        
        return stack[0]
    
    @staticmethod
    def daily_temperatures(temperatures: List[int]) -> List[int]:
        """
        Daily Temperatures - Monotonic stack
        Time: O(n), Space: O(n)
        """
        n = len(temperatures)
        result = [0] * n
        stack = []
        
        for i in range(n):
            while stack and temperatures[stack[-1]] < temperatures[i]:
                prev_index = stack.pop()
                result[prev_index] = i - prev_index
            stack.append(i)
        
        return result
    
    @staticmethod
    def car_fleet(target: int, position: List[int], speed: List[int]) -> int:
        """
        Car Fleet - Monotonic stack
        Time: O(n log n), Space: O(n)
        """
        cars = sorted(zip(position, speed), reverse=True)
        stack = []
        
        for pos, spd in cars:
            time_to_target = (target - pos) / spd
            
            if not stack or time_to_target > stack[-1]:
                stack.append(time_to_target)
        
        return len(stack)


# Testing and Benchmarking

def test_solutions():
    """Test function to verify solutions work correctly"""
    
    # Test cases
    test_cases = [
        {
            "name": "Next Greater Element",
            "input": [4, 5, 2, 10],
            "expected": [5, 10, 10, -1]
        },
        {
            "name": "Valid Parentheses",
            "input": "()[]{}",
            "expected": True
        },
        {
            "name": "Daily Temperatures",
            "input": [73, 74, 75, 71, 69, 72, 76, 73],
            "expected": [1, 1, 4, 2, 1, 1, 0, 0]
        }
    ]
    
    patterns = StackPatterns()
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        
        if test['name'] == "Next Greater Element":
            result = patterns.next_greater_element(test['input'])
        elif test['name'] == "Valid Parentheses":
            result = patterns.valid_parentheses(test['input'])
        elif test['name'] == "Daily Temperatures":
            result = patterns.daily_temperatures(test['input'])
        
        print(f"Input: {test['input']}")
        print(f"Expected: {test['expected']}")
        print(f"Result: {result}")
        print(f"Pass: {result == test['expected']}")


def benchmark_solutions():
    """Benchmark different approaches to compare performance"""
    
    # Example: Compare different approaches for the same problem
    nums = list(range(10000))
    
    print("Benchmarking Stack approaches:")
    
    # Approach 1: Brute Force
    start_time = time.time()
    # brute_force_result = brute_force_next_greater(nums)
    brute_force_time = time.time() - start_time
    
    # Approach 2: Stack
    start_time = time.time()
    stack_result = StackPatterns.next_greater_element(nums)
    stack_time = time.time() - start_time
    
    print(f"Brute Force: {brute_force_time:.6f} seconds")
    print(f"Stack: {stack_time:.6f} seconds")
    print(f"Speedup: {brute_force_time / stack_time:.2f}x")


if __name__ == "__main__":
    print("Stack Solutions - NeetCode Style")
    print("=" * 50)
    
    # Run tests
    test_solutions()
    
    # Run benchmarks
    print("\n" + "=" * 50)
    benchmark_solutions()
    
    print("\nTemplate ready for use! Add your specific problem solutions here.")
