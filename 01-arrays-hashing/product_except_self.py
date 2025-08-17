"""
Product of Array Except Self - LeetCode Problem 238
https://leetcode.com/problems/product-of-array-except-self/

Given an integer array nums, return an array answer such that answer[i] is equal to 
the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

Example 1:
Input: nums = [1,2,3,4]
Output: [24,12,8,6]
Explanation: answer[0] = 3*4*2 = 24, answer[1] = 1*3*4 = 12, answer[2] = 1*2*4 = 8, answer[3] = 1*2*3 = 6

Example 2:
Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
Explanation: answer[0] = 1*0*(-3)*3 = 0, answer[1] = (-1)*0*(-3)*3 = 0, answer[2] = (-1)*1*(-3)*3 = 9, answer[3] = (-1)*1*0*3 = 0, answer[4] = (-1)*1*0*(-3) = 0

Constraints:
- 2 <= nums.length <= 10^5
- -30 <= nums[i] <= 30
- The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
"""

from typing import List
import time


class Solution:
    """
    Solution class with multiple approaches to solve Product of Array Except Self
    """
    
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """
        Main method - returns the most efficient solution
        """
        return self.product_except_self_prefix_suffix(nums)
    
    def product_except_self_prefix_suffix(self, nums: List[int]) -> List[int]:
        """
        Approach 1: Prefix and Suffix Arrays - Optimal Solution
        
        Algorithm:
        1. Create prefix array where prefix[i] = product of all elements before i
        2. Create suffix array where suffix[i] = product of all elements after i
        3. Result[i] = prefix[i] * suffix[i]
        
        Time Complexity: O(n) - Three passes through the array
        Space Complexity: O(n) - Need prefix and suffix arrays
        
        Analysis:
        - Pros: Clear logic, easy to understand
        - Cons: Uses extra space for prefix and suffix arrays
        """
        n = len(nums)
        
        # Calculate prefix products
        prefix = [1] * n
        for i in range(1, n):
            prefix[i] = prefix[i - 1] * nums[i - 1]
        
        # Calculate suffix products
        suffix = [1] * n
        for i in range(n - 2, -1, -1):
            suffix[i] = suffix[i + 1] * nums[i + 1]
        
        # Calculate result
        result = [prefix[i] * suffix[i] for i in range(n)]
        
        return result
    
    def product_except_self_constant_space(self, nums: List[int]) -> List[int]:
        """
        Approach 2: Constant Space - Most Efficient
        
        Algorithm:
        1. Use result array to store prefix products
        2. Use a variable to track suffix product
        3. Update result array in reverse order
        
        Time Complexity: O(n) - Two passes through the array
        Space Complexity: O(1) - Only using result array (not counting output)
        
        Analysis:
        - Pros: Optimal space complexity, very efficient
        - Cons: Slightly more complex logic
        """
        n = len(nums)
        result = [1] * n
        
        # Calculate prefix products and store in result
        for i in range(1, n):
            result[i] = result[i - 1] * nums[i - 1]
        
        # Calculate suffix products and multiply with prefix
        suffix = 1
        for i in range(n - 1, -1, -1):
            result[i] = result[i] * suffix
            suffix *= nums[i]
        
        return result
    
    def product_except_self_division(self, nums: List[int]) -> List[int]:
        """
        Approach 3: Using Division (Not allowed in problem but for comparison)
        
        Algorithm:
        1. Calculate total product of all elements
        2. For each element, divide total product by that element
        
        Time Complexity: O(n) - Two passes through the array
        Space Complexity: O(1) - Only using result array
        
        Analysis:
        - Pros: Simple logic, efficient
        - Cons: Uses division (not allowed), doesn't handle zeros well
        """
        n = len(nums)
        total_product = 1
        zero_count = 0
        
        # Count zeros and calculate product of non-zero elements
        for num in nums:
            if num == 0:
                zero_count += 1
            else:
                total_product *= num
        
        result = [0] * n
        
        if zero_count > 1:
            # If more than one zero, all products are zero
            return result
        elif zero_count == 1:
            # If exactly one zero, only that position has non-zero product
            for i in range(n):
                if nums[i] == 0:
                    result[i] = total_product
                    break
        else:
            # No zeros, use division
            for i in range(n):
                result[i] = total_product // nums[i]
        
        return result
    
    def product_except_self_brute_force(self, nums: List[int]) -> List[int]:
        """
        Approach 4: Brute Force - Check all pairs (for educational purposes)
        
        Algorithm:
        1. For each element, calculate product of all other elements
        2. Use nested loops to skip current element
        
        Time Complexity: O(n²) - For each element, multiply with n-1 other elements
        Space Complexity: O(1) - Only using result array
        
        Analysis:
        - Pros: Simple to understand
        - Cons: Very inefficient, quadratic time complexity
        """
        n = len(nums)
        result = [1] * n
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    result[i] *= nums[j]
        
        return result
    
    def product_except_self_logarithm(self, nums: List[int]) -> List[int]:
        """
        Approach 5: Using Logarithms (Advanced)
        
        Algorithm:
        1. Take log of each element
        2. Calculate sum of all logs
        3. For each element, subtract its log from total sum
        4. Take exponential to get final result
        
        Time Complexity: O(n) - Single pass through the array
        Space Complexity: O(1) - Only using result array
        
        Analysis:
        - Pros: Single pass solution
        - Cons: Floating point precision issues, complex
        """
        import math
        
        n = len(nums)
        
        # Handle zeros and negative numbers
        zero_count = nums.count(0)
        
        if zero_count > 1:
            return [0] * n
        elif zero_count == 1:
            result = [0] * n
            zero_index = nums.index(0)
            product = 1
            for i in range(n):
                if i != zero_index:
                    product *= nums[i]
            result[zero_index] = product
            return result
        
        # Calculate sum of logarithms
        log_sum = sum(math.log(abs(num)) for num in nums)
        
        result = []
        for num in nums:
            # Calculate product except self using logarithms
            product_log = log_sum - math.log(abs(num))
            product = round(math.exp(product_log))
            
            # Handle sign
            negative_count = sum(1 for x in nums if x < 0)
            current_negative = 1 if num < 0 else 0
            if (negative_count - current_negative) % 2 == 1:
                product = -product
            
            result.append(product)
        
        return result


# Testing and Benchmarking

def test_product_except_self():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "nums": [1, 2, 3, 4],
            "expected": [24, 12, 8, 6],
            "description": "Basic case"
        },
        {
            "nums": [-1, 1, 0, -3, 3],
            "expected": [0, 0, 9, 0, 0],
            "description": "With zeros"
        },
        {
            "nums": [2, 3],
            "expected": [3, 2],
            "description": "Two elements"
        },
        {
            "nums": [1, 1, 1, 1],
            "expected": [1, 1, 1, 1],
            "description": "All ones"
        },
        {
            "nums": [0, 0],
            "expected": [0, 0],
            "description": "All zeros"
        },
        {
            "nums": [-2, -3, -4],
            "expected": [12, 8, 6],
            "description": "All negative"
        }
    ]
    
    approaches = [
        ("Prefix/Suffix", solution.product_except_self_prefix_suffix),
        ("Constant Space", solution.product_except_self_constant_space),
        ("Division", solution.product_except_self_division),
        ("Logarithm", solution.product_except_self_logarithm)
    ]
    
    print("Product of Array Except Self - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: nums = {test['nums']}")
        print(f"Expected: {test['expected']}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(test['nums'])
                
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
    nums = [1] * n  # All ones for simple calculation
    
    print(f"\nBenchmarking with array of size {n}")
    print("=" * 50)
    
    approaches = [
        ("Prefix/Suffix", solution.product_except_self_prefix_suffix),
        ("Constant Space", solution.product_except_self_constant_space),
        ("Division", solution.product_except_self_division)
    ]
    
    # Skip brute force and logarithm for large input
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(nums)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds")
        print(f"  First few results: {result[:5]}")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Prefix/Suffix", "O(n)", "O(n)", "Clear logic, extra space"),
        ("Constant Space", "O(n)", "O(1)", "Optimal space complexity"),
        ("Division", "O(n)", "O(1)", "Simple but uses division"),
        ("Logarithm", "O(n)", "O(1)", "Single pass, precision issues"),
        ("Brute Force", "O(n²)", "O(1)", "Educational only")
    ]
    
    print(f"{'Approach':<20} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<20} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Product of Array Except Self - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_product_except_self()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use constant space approach for optimal performance.")
