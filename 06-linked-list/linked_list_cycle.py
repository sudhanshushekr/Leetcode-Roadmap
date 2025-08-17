"""
Linked List Cycle - LeetCode Problem 141
https://leetcode.com/problems/linked-list-cycle/

Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached 
again by continuously following the next pointer. Internally, pos is used to denote the 
index of the node that tail's next pointer is connected to. Note that pos is not passed 
as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

Example 1:
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

Example 2:
Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.

Example 3:
Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.

Constraints:
- The number of the nodes in the list is in the range [0, 10^4].
- -10^5 <= Node.val <= 10^5
- pos is -1 or a valid index in the linked-list.
"""

from typing import Optional
import time


class ListNode:
    """Definition for singly-linked list node"""
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    """
    Solution class with multiple approaches to solve Linked List Cycle
    """
    
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """
        Main method - returns the most efficient solution
        """
        return self.has_cycle_fast_slow_pointers(head)
    
    def has_cycle_hash_set(self, head: Optional[ListNode]) -> bool:
        """
        Approach 1: Hash Set - Track visited nodes
        
        Algorithm:
        1. Use a hash set to track nodes we've visited
        2. Traverse the linked list
        3. If we encounter a node that's already in the set, there's a cycle
        4. If we reach the end (None), there's no cycle
        
        Time Complexity: O(n) - We visit each node at most once
        Space Complexity: O(n) - Hash set can store up to n nodes
        
        Analysis:
        - Pros: Simple to understand and implement
        - Cons: Uses extra space for hash set
        """
        visited = set()
        current = head
        
        while current:
            if current in visited:
                return True
            visited.add(current)
            current = current.next
        
        return False
    
    def has_cycle_fast_slow_pointers(self, head: Optional[ListNode]) -> bool:
        """
        Approach 2: Fast and Slow Pointers (Floyd's Cycle Detection)
        
        Algorithm:
        1. Use two pointers: slow (moves 1 step) and fast (moves 2 steps)
        2. If there's a cycle, fast will eventually catch up to slow
        3. If there's no cycle, fast will reach the end (None)
        
        Time Complexity: O(n) - In worst case, fast pointer traverses the list twice
        Space Complexity: O(1) - Only using two pointers
        
        Analysis:
        - Pros: Optimal space complexity, very efficient
        - Cons: Slightly more complex logic
        """
        if not head or not head.next:
            return False
        
        slow = head
        fast = head.next
        
        while slow != fast:
            if not fast or not fast.next:
                return False
            slow = slow.next
            fast = fast.next.next
        
        return True
    
    def has_cycle_brute_force(self, head: Optional[ListNode]) -> bool:
        """
        Approach 3: Brute Force - Check each node against all previous nodes
        
        Algorithm:
        1. For each node, check if it points to any previous node
        2. If found, there's a cycle
        3. If we reach the end, there's no cycle
        
        Time Complexity: O(n²) - For each node, check all previous nodes
        Space Complexity: O(1) - Only using pointers
        
        Analysis:
        - Pros: Simple logic, no extra space
        - Cons: Very inefficient, quadratic time complexity
        """
        if not head:
            return False
        
        current = head.next
        position = 1
        
        while current:
            # Check if current node points to any previous node
            check = head
            for i in range(position):
                if current == check:
                    return True
                check = check.next
            
            current = current.next
            position += 1
        
        return False
    
    def has_cycle_mark_visited(self, head: Optional[ListNode]) -> bool:
        """
        Approach 4: Mark Visited Nodes (Modifies the list)
        
        Algorithm:
        1. Mark each visited node by setting a special value
        2. If we encounter a marked node, there's a cycle
        3. If we reach the end, there's no cycle
        
        Time Complexity: O(n) - We visit each node at most once
        Space Complexity: O(1) - No extra space needed
        
        Analysis:
        - Pros: No extra space, simple logic
        - Cons: Modifies the original list (not always acceptable)
        """
        if not head:
            return False
        
        current = head
        while current:
            if current.val == float('inf'):  # Mark as visited
                return True
            current.val = float('inf')  # Mark current node
            current = current.next
        
        return False
    
    def has_cycle_reverse_pointers(self, head: Optional[ListNode]) -> bool:
        """
        Approach 5: Reverse Pointers (Advanced)
        
        Algorithm:
        1. Reverse the pointers as we traverse
        2. If we can reach the head again, there's a cycle
        3. If we reach None, there's no cycle
        
        Time Complexity: O(n) - We visit each node at most once
        Space Complexity: O(1) - Only using pointers
        
        Analysis:
        - Pros: No extra space, clever approach
        - Cons: Destroys the original list structure
        """
        if not head or not head.next:
            return False
        
        prev = None
        current = head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
            
            # If we can reach head again, there's a cycle
            if current == head:
                return True
        
        return False


# Helper functions for testing

def create_linked_list_with_cycle(values: list, pos: int) -> Optional[ListNode]:
    """Create a linked list with cycle for testing"""
    if not values:
        return None
    
    # Create nodes
    nodes = []
    for val in values:
        nodes.append(ListNode(val))
    
    # Connect nodes
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    
    # Create cycle if pos is valid
    if pos >= 0 and pos < len(nodes):
        nodes[-1].next = nodes[pos]
    
    return nodes[0]


def linked_list_to_list(head: Optional[ListNode]) -> list:
    """Convert linked list to list (for testing, limited to avoid infinite loops)"""
    result = []
    current = head
    count = 0
    max_count = 100  # Prevent infinite loops
    
    while current and count < max_count:
        result.append(current.val)
        current = current.next
        count += 1
    
    return result


# Testing and Benchmarking

def test_linked_list_cycle():
    """Test all approaches with various test cases"""
    
    solution = Solution()
    
    test_cases = [
        {
            "values": [3, 2, 0, -4],
            "pos": 1,
            "expected": True,
            "description": "Cycle exists"
        },
        {
            "values": [1, 2],
            "pos": 0,
            "expected": True,
            "description": "Cycle to first node"
        },
        {
            "values": [1],
            "pos": -1,
            "expected": False,
            "description": "Single node, no cycle"
        },
        {
            "values": [],
            "pos": -1,
            "expected": False,
            "description": "Empty list"
        },
        {
            "values": [1, 2, 3, 4, 5],
            "pos": 2,
            "expected": True,
            "description": "Cycle to middle node"
        },
        {
            "values": [1, 2, 3, 4, 5],
            "pos": -1,
            "expected": False,
            "description": "No cycle"
        }
    ]
    
    approaches = [
        ("Hash Set", solution.has_cycle_hash_set),
        ("Fast/Slow Pointers", solution.has_cycle_fast_slow_pointers),
        ("Brute Force", solution.has_cycle_brute_force),
        ("Mark Visited", solution.has_cycle_mark_visited),
        ("Reverse Pointers", solution.has_cycle_reverse_pointers)
    ]
    
    print("Linked List Cycle - Testing All Approaches")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nTest Case: {test['description']}")
        print(f"Input: values = {test['values']}, pos = {test['pos']}")
        print(f"Expected: {test['expected']}")
        
        # Create linked list for this test
        head = create_linked_list_with_cycle(test['values'], test['pos'])
        
        for approach_name, approach_func in approaches:
            try:
                # Create a fresh copy for each test to avoid modifications
                test_head = create_linked_list_with_cycle(test['values'], test['pos'])
                result = approach_func(test_head)
                
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
    values = list(range(n))
    pos = n // 2  # Cycle to middle
    
    head = create_linked_list_with_cycle(values, pos)
    
    print(f"\nBenchmarking with linked list of size {n}")
    print("=" * 50)
    
    approaches = [
        ("Hash Set", solution.has_cycle_hash_set),
        ("Fast/Slow Pointers", solution.has_cycle_fast_slow_pointers)
    ]
    
    # Skip brute force for large input as it's too slow
    for approach_name, approach_func in approaches:
        start_time = time.time()
        result = approach_func(head)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{approach_name}: {execution_time:.6f} seconds (Result: {result})")


def complexity_analysis():
    """Print complexity analysis for all approaches"""
    
    print("\nComplexity Analysis")
    print("=" * 50)
    
    analysis = [
        ("Hash Set", "O(n)", "O(n)", "Simple, uses extra space"),
        ("Fast/Slow Pointers", "O(n)", "O(1)", "Optimal, most efficient"),
        ("Brute Force", "O(n²)", "O(1)", "Simple but very slow"),
        ("Mark Visited", "O(n)", "O(1)", "Modifies list"),
        ("Reverse Pointers", "O(n)", "O(1)", "Destroys list structure")
    ]
    
    print(f"{'Approach':<20} {'Time':<10} {'Space':<10} {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in analysis:
        print(f"{approach:<20} {time_comp:<10} {space_comp:<10} {notes}")


if __name__ == "__main__":
    print("Linked List Cycle - Complete Solution Analysis")
    print("=" * 60)
    
    # Run tests
    test_linked_list_cycle()
    
    # Run benchmarks
    benchmark_approaches()
    
    # Show complexity analysis
    complexity_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed! Use the fast/slow pointers approach for optimal performance.")
