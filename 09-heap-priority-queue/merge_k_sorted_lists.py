"""
Merge k Sorted Lists - LeetCode Problem 23
https://leetcode.com/problems/merge-k-sorted-lists/

You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

Example 1:
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: merging the above 3 lists produces [1,1,2,3,4,4,5,6].

Example 2:
Input: lists = []
Output: []

Example 3:
Input: lists = [[]]
Output: []

Constraints:
- k == lists.length
- 0 <= k <= 10^4
- 0 <= lists[i].length <= 500
- -10^4 <= lists[i][j] <= 10^4
- lists[i] is sorted in ascending order.
"""

from typing import List, Optional
import heapq
import time


class ListNode:
    """Definition for singly-linked list node"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
    Solution class with multiple approaches to solve Merge K Sorted Lists
    """
    
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Main method - returns the most efficient solution
        """
        return self.merge_k_lists_heap(lists)
    
    def merge_k_lists_heap(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Approach 1: Min Heap (Priority Queue) - Most Efficient
        
        Algorithm:
        1. Use a min heap to keep track of the smallest element from each list
        2. Extract the minimum element and add it to result
        3. Add the next element from the same list to heap
        4. Repeat until heap is empty
        
        Time Complexity: O(n log k) - n total nodes, k lists
        Space Complexity: O(k) - Heap size
        
        Analysis:
        - Pros: Optimal time complexity, handles any number of lists
        - Cons: Uses extra space for heap
        """
        if not lists:
            return None
        
        # Create min heap with (value, list_index, node)
        heap = []
        for i, head in enumerate(lists):
            if head:
                heapq.heappush(heap, (head.val, i, head))
        
        dummy = ListNode(0)
        current = dummy
        
        while heap:
            val, list_idx, node = heapq.heappop(heap)
            current.next = ListNode(val)
            current = current.next
            
            # Add next node from the same list
            if node.next:
                heapq.heappush(heap, (node.next.val, list_idx, node.next))
        
        return dummy.next
    
    def merge_k_lists_divide_conquer(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Approach 2: Divide and Conquer - Merge pairs
        
        Algorithm:
        1. Merge lists in pairs
        2. Continue until only one list remains
        3. Use merge two sorted lists as helper
        
        Time Complexity: O(n log k) - n total nodes, k lists
        Space Complexity: O(log k) - Recursion stack
        
        Analysis:
        - Pros: Good time complexity, no extra data structures
        - Cons: More complex implementation
        """
        if not lists:
            return None
        
        def merge_two_lists(l1, l2):
            dummy = ListNode(0)
            current = dummy
            
            while l1 and l2:
                if l1.val <= l2.val:
                    current.next = l1
                    l1 = l1.next
                else:
                    current.next = l2
                    l2 = l2.next
                current = current.next
            
            current.next = l1 if l1 else l2
            return dummy.next
        
        def merge_lists_recursive(lists, start, end):
            if start == end:
                return lists[start]
            if start > end:
                return None
            
            mid = (start + end) // 2
            left = merge_lists_recursive(lists, start, mid)
            right = merge_lists_recursive(lists, mid + 1, end)
            
            return merge_two_lists(left, right)
        
        return merge_lists_recursive(lists, 0, len(lists) - 1)
    
    def merge_k_lists_brute_force(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Approach 3: Brute Force - Collect all nodes and sort
        
        Algorithm:
        1. Collect all nodes from all lists
        2. Sort them by value
        3. Create a new linked list
        
        Time Complexity: O(n log n) - Sort all nodes
        Space Complexity: O(n) - Store all nodes
        
        Analysis:
        - Pros: Simple to implement
        - Cons: Inefficient, doesn't use sorted property
        """
        if not lists:
            return None
        
        # Collect all nodes
        nodes = []
        for head in lists:
            current = head
            while current:
                nodes.append(current.val)
                current = current.next
        
        # Sort nodes
        nodes.sort()
        
        # Create new linked list
        if not nodes:
            return None
        
        dummy = ListNode(0)
        current = dummy
        for val in nodes:
            current.next = ListNode(val)
            current = current.next
        
        return dummy.next
    
    def merge_k_lists_compare_one_by_one(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Approach 4: Compare One by One - Simple but inefficient
        
        Algorithm:
        1. Find the minimum value among all list heads
        2. Add it to result and move that list pointer
        3. Repeat until all lists are exhausted
        
        Time Complexity: O(k * n) - k lists, n total nodes
        Space Complexity: O(1) - No extra space
        
        Analysis:
        - Pros: Simple logic, no extra space
        - Cons: Very inefficient for large k
        """
        if not lists:
            return None
        
        dummy = ListNode(0)
        current = dummy
        
        while True:
            min_val = float('inf')
            min_idx = -1
            
            # Find minimum value among all list heads
            for i, head in enumerate(lists):
                if head and head.val < min_val:
                    min_val = head.val
                    min_idx = i
            
            # If no more elements, break
            if min_idx == -1:
                break
            
            # Add minimum element to result
            current.next = ListNode(min_val)
            current = current.next
            
            # Move pointer in the list with minimum element
            lists[min_idx] = lists[min_idx].next
        
        return dummy.next


# Helper functions for testing

def create_linked_list(values: list) -> Optional[ListNode]:
    """Create a linked list from a list of values"""
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head


def linked_list_to_list(head: Optional[ListNode]) -> list:
    """Convert linked list to list for testing"""
    result = []
    current = head
    
    while current:
        result.append(current.val)
        current = current.next
    
    return result


# Testing and Benchmarking

def test_merge_k_sorted_lists():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "lists": [[1, 4, 5], [1, 3, 4], [2, 6]],
            "expected": [1, 1, 2, 3, 4, 4, 5, 6],
            "description": "Three sorted lists"
        },
        {
            "lists": [],
            "expected": [],
            "description": "Empty list"
        },
        {
            "lists": [[]],
            "expected": [],
            "description": "Single empty list"
        },
        {
            "lists": [[1], [2], [3]],
            "expected": [1, 2, 3],
            "description": "Single element lists"
        },
        {
            "lists": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "expected": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "description": "Sequential lists"
        }
    ]
    
    approaches = [
        ("Min Heap", solution.merge_k_lists_heap),
        ("Divide & Conquer", solution.merge_k_lists_divide_conquer),
        ("Brute Force", solution.merge_k_lists_brute_force),
        ("Compare One by One", solution.merge_k_lists_compare_one_by_one)
    ]
    
    print("Merge K Sorted Lists - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: lists = {test['lists']}")
        print(f"Expected: {test['expected']}")
        
        # Create linked lists for this test
        linked_lists = [create_linked_list(lst) for lst in test['lists']]
        
        for approach_name, approach_func in approaches:
            try:
                result_head = approach_func(linked_lists)
                result = linked_list_to_list(result_head)
                
                passed = result == test['expected']
                status = "✅ PASS" if passed else "❌ FAIL"
                
                print(f"  {approach_name}: {result} {status}")
                
            except Exception as e:
                print(f"  {approach_name}: ERROR - {e}")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Min Heap", "O(n log k)", "O(k)", "Optimal time complexity"),
        ("Divide & Conquer", "O(n log k)", "O(log k)", "Good balance"),
        ("Brute Force", "O(n log n)", "O(n)", "Simple but inefficient"),
        ("Compare One by One", "O(k * n)", "O(1)", "Simple but very slow")
    ]
    
    print(f"{'Approach':<20} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<20} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Merge K Sorted Lists - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_merge_k_sorted_lists()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the min heap approach for optimal performance.")
