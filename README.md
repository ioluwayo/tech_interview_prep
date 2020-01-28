### Interview prep summary

``` 
maxint = sys.maxsize
min = -sys.maxsize
```




The goal is 20 leetcode questions in 1 week. 15 easy 5 medium

1. Find the majority element in an array [https://leetcode.com/problems/majority-element/]
    So any number that appears more than n/2 times is the answer. We can just count how often each item happpens 
    and return the first that is greater than n/2 times. 
    
    Appearing more than n/2 times means it always takes up more than half of the array.     
    
    Boyers voting algorithm of counting majority/minority items using +/-1
    
    1. For each element count its occurence and check if its > n/2. 
    
    O(n^2) O(1)
    ```
    def majorityElement(self, nums):
        majority_count = len(nums)//2
        for num in nums:
            count = sum(1 for elem in nums if elem == num)
            if count > majority_count:
                return num
                
    ```
     
    2. Sort and return item at n/2 location (also n/2 + 1  if n is even)
    
    O(nlog(n)) O(1)
    ```
    def majorityElement(self, nums):
        nums.sort()
        return nums[len(nums)//2]
    ```
        
    3. Use a hashmap to count and return element with largest count. You can avoid 2 loops by tracking the max while 
    buidling the count_map.
    
    O(n) O(n)
    
    ```
    def majorityElement(self, nums):
        counts = collections.Counter(nums)
        return max(counts.keys(), key=counts.get) # major! the comparison is done on count.get(k) but the k is returned.
    ```
    
    4. Randomly select an item, check its count in the array. If its more than n/2 then return.
    
    O(inf) but bounded still cause the item appears n/2 times. O(1)
    ```
    def majorityElement(self, nums):
        majority_count = len(nums)//2
        while True:
            candidate = random.choice(nums)
            if sum(1, for elem in nums if elem == candidate) > majority_count
                return candidate
    ```
    
    5. Boyer-Moore Voting algorithm. 
    
    Imagine assuming every element is the majority element at first. If it is, add 1 to a count if it is not add -1 
    to the count.
    You know it is not whenever the count reaches 0 so you start again by assuming the current item is the majority!
    
    [7, 7, 5, 7, 5, 1 | 5, 7 | 5, 5, 7, 7 | 7, 7, 7, 7]
    count becomes 0 at evry |
    
    We are unable to reach negative count becuase it is imposible to have more none majority items than majority items!
    
    [7, 7, 5, 7, 5, 1 | 5, 7 | 5, 5, 7, 7 | 5, 5, 5, 5]
    
    O(n) O(1)
    ```
    def majorityElement(self, nums):
        count = 0
        candidate = nums[0]
        for i in nums:
            if count == 0:
                candidate = i
           count += 1 if i==candidate else -1
        return candidate
    ```
2. Valid anagram [https://leetcode.com/problems/valid-anagram/]
    
    Consider strings s and t. An anagram is produced by rearranging the letters of s into t. s and t must be the same 
    lenght and contain the exact same characters and frequencies.
    
    1. We can sort both string and compare them. They are anagarams if equal!
    
    O(nlogn) O(n)
    ```
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        s = sorted(s)
        t = sorted(t)
        return t == s
    # string comparison is O(n). sorting time dominates so overall is O(nlogn).
    # using sorted() requires copying the array. This is implementation specific so could be O(1)
    ```
    2. We can maintain frequency count of the characters. We increment the freqnecy for s and decrement it. If we 
    ever reach a frequency of -1 for any character they are not anagrams.
    
    O(n) O(1)
    ```
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        count = [0 for i in range(26)]
        a = ord('a') # this is how we convert a to its ascii numeric representation. this way we dont need a hashmap
        for i in range(len(s)):
            count[ord(s[i])-a]+=1
            count[ord(t[i])-a]-=1
        for i in count:
            if i < 0:
                return False
        return True
    ```
   
    3. If the problem included non ascii characters, then we need an array of size 1114112. This is too large!
    we will need to use a hashmap so we only use memory for characters that exist in the string 
 
    O(n) O(n)
    ```
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        count_map = {}
        for i in range(len(s)):
            count_map[s[i]] = count_map.get(s[i], 0)+1
            count_map[t[i]] = count_map.get(t[i], 0)-1
        for i in count_map.values():
            if i < 0:
                return False
        return True
    ```
3. Contains duplicate [https://leetcode.com/problems/contains-duplicate/]
    
    With O(n^2) we can search for duplicate occurence of eahc item in the list. 
    
    1. A better approach is to sort the list and then check neighboring elems.
    
    O(nlogn), O(1)
    ```
    a.sort()
    for i in range(len(a)-1):
        if a[i] == a[i+1]:
            return True
        return False
   ```
   2. The best approach is to use a set to store the items. This gives O(1) search time and O(1) insertion time. Unlike 
   a self balancing tree which would be O(log n) for both operations. 
   
   O(n), O(n)
   ```
   def containsDuplicate(self, nums: List[int]) -> bool:
    nums_set = set()
    for i in nums:
        if i in nums_set:
            return True
        nums_set.add(i)
    return False
   
   ```
4. Roman to integer [https://leetcode.com/problems/roman-to-integer/]
    We define a value map for roman numerals and then loop from behind adding up the values.
    If the current value is smaller than the previous then we subtract it form the total.
    
    O(n)m O(1)
    ```
    def romanToInt(self, s: str) -> int:
        value_map = {
            "I": 1,
            "V": 5, 
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000
        }
        prev = value_map[s[-1]]
        total = prev
        for i in reversed(range(len(s)-1)):
            val = value_map[s[i]]
            if val < prev:
                total-=val
            else:
                total+=val
            prev = val
        return total
    
    ```
5. Best time to buy and sell stock II [https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/solution/]
    The key here is that we are allowed to buy and sell on the same day. This simplifies the solution alot
    The only restriction is that we wave to sell before buying again. We cannot buy or sell 2 times in a row.
    
    Very simple!!
    The best approach is just to buy and sell whenever there is a profit! The freedom to buy and sell on the same day
    makes this possible. Skipping any possible profit will always lead to less total profit at the end.
    
    O(n), O(1)
    
    ``` 
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        if prices:
            prev = prices[0]
            for i in range(1, len(prices)):
                curr = prices[i]
                if prev < curr:
                    max_profit += curr-prev
                prev = curr
        return max_profit
    ```

6. Convert sorted array to Binary Search Tree [https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/]
    For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.
    1. The recursize approach is quite simple and staright forward.
        1. Imagine 2 base cases. A list with 3 items, and a list with no item.
        2. A empty list should simply return None.
        3. A list with 3 items will have one parent and 2 children.
        4. Identifiy the middle index.
        5. Create a node with that index.
        4. Now we need to call the function again on the left and right halves to do the same thing and return its own 
        mid node when done.
        5. Now, test the logic with a list with just 1 item. both left and right nodes will be set to None as expected
        
        O(n), O(n) recursive stack of depth n
        ``` 
        def sortedArrayToBST(sum, nums: List[int]) -> TreeNode:
            if not nums: return None
            left = 0
            length = len(nums)
            mid = (left + length)//2 
            node = TreeNode(nums[mid])
            node.left = self.sortedArrayToBST(nums[start:mid]) # nums[:mid]
            node.right = self.sortedArrayToBST(nums[mid+1:length]) # nums[mid+1:]
            return node
        ```
    2. Iterative solution??
    
7. First unique character. [https://leetcode.com/problems/first-unique-character-in-a-string/]
    
    Best approach is to build a count map in O(n) time and then loop though the string one more time checking for the
     first count of 1.
     
     O(n), O(n)
     ``` 
     def firstUniqChar(self, s: str) -> int:
        count_map = collections.Counter(s)
        for i in range(len(s)):
            if count_map[s[i]] == 1: return i
        return -1
     ```
8.  Missing number [https://leetcode.com/problems/missing-number/]
    Pay attentin to what the description of this question. At first glance you may not notice some key details that 
    will help in solving the problem using XOR
    
    The array contains n distinct numbers. ie 10 distinct numbers. The numbers are take from 0, 1, 2, ...., n finf 
    the one that is missing from the array.
    
    let n be 5
    so array wil be [0, 1, 2, 3, 5] or [0, 2, 3, 4, 5], or [1, 2, 4, 5, 3] (array may not be sorted)
    
    1. The array is always of size n but it would be of size n+1 had the missing number be there.
    2. The array should contain all the numbers from 0-n idealy. This would lead to a lenght of n+1!
    [0, 1, 2, 3, 4 ,5] but instead on is missing so [1, 2, 3, 4, 5] or [0, 1, 2, 3, 4]
    The question here is what is missing?
    
    3. Pay attention to the index of all the numbers
    
    Most obvious approahc is to sort and just check for the missing item. We know all the numbers that should have 
    been there. 0-n!
    
    O(n), O(1)
    ``` 
    def missingNumber(self, nums: List[int]) -> int:
        nums.sort()
        count = 0 # start counting from 0 to n
        for i in nums:
            if i != count:
                return count
            count += 1
        return count        
    ```
    Another nice approach is to creat a hashset of the numbers in the array. This makes it possible to search of 
    items in constant time. Rather than searching the original array
    
    O(n), O(n)
    ``` 
    def missingNumber(self, nums: List[int]) -> int:
        nums_set = set(nums)
        count = 0 # start counting from 0 to n
        for i in range(len(nums+1)):
            if i != nums_set:
                return i
               
    ```
    
    The most interesteing soluttion which is to use XOR. At first glance you might not see how. 
    The trick is to use the damn indeces!
    say arr is [0, 1, 2, 3, 5]. The missing guy is 5. You might wonder how to use the whole exclusive idea ay first.
    
    arr[0]: 0 <br>
    arr[1]: 1 <br>
    arr[2]: 2 <br>
    arr[3]: 3 <br>
    arr[4]: 5 <br>
    
    If you consider both the indeces and the values, all the numbers occur 2 times! err the missinf number and n do 
    not!<br>
    But they do in the ideal array!
    
    arr[0]: 0 <br>
    arr[1]: 1 <br>
    arr[2]: 2 <br>
    arr[3]: 3 <br>
    arr[4]: 5 <br>
    arr[5]: 4 <br>
    So what we will do here is to include n ourselves! and then XOR the whole lot! <br>
    Including n ourselves is still fine, because if it was the missing guy, The one we inserted will be the only one.
    So it is stil exclusive. If it was not the missing guy, we have introduced it matching pair. The array never 
    comes with it, so we need to introduce it for the solution to work. n is a missing index, and our answer is a 
    missing value!!
    
    O(n), O(1)        
    ``` 
    def missingNumber(self, nums: List[int]) -> int:
        ans = len(nums)
        for i in range(len(nums)):
            ans = ans ^ i ^ nums[i]
        return ans
    ```
     
    
    
    
9. In order traversal successor in BST [https://leetcode.com/problems/inorder-successor-in-bst/submissions/]
   We want the Node after the target node when a binary tree is present inorder. This is technically the node 
   immediately greater that the querry node.
   
   Off the Bat we can just do in order traversal and store the result in a list then search for the target and return
    the node after it.
    
    O(H), O(n), where H is the height of the tree
    
    ``` 
    class Solution:
        result = []
        def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
            self.inorder(root)
            for i in range(len(self.result)-1):
                if self.result[i] == p:
                    return self.result[i+1]
            return None
        def inorder(self, root):
            if not root:
                return
            self.inorder(root.left)
            self.result.append(root)
            self.inorder(root.right)
    ```
    
    We are actually able to do this without any extra space. We just need to notice the following.
    
    1. If the target node has a right subtree, the sucessor is the minimum element in that sub tree.
    2. Else, the successor is its most recent ancestor for which its a left child.

   ```
   
   class Solution:
        result = []
        def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
            last_left_anc = None
            while root:
                if p == root:
                    if root.right:
                        return self.minimum(root.right)
                    else:
                        return last_left_anc
                else:
                    if p.val < root.val:
                        last_left_anc = root
                        root = root.left
                    else:
                        root = root.right
            
        def minimum(self, root):
            while root:
                if root.left:
                    root = root.left
                else:
                    break
        
            return root
       
   ```
    
   
We can do better tho. Notice that the successor is greater than the target
so we can update the succ value whenever we encounter a value greater than target. We keep progressing towards 
the minimum value greater than the target.
   
   
O(H), O(1)

``` 
  class Solution:
    result = []
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        succ = None
        while root:
            if p.val<root.val:
                succ = root # update it anytime you go left because thats closer to p in inorder
                root = root.left
            else:
                root = root.right
        return succ
```
    
    
10. Merge sorted linked list [https://leetcode.com/problems/merge-two-sorted-lists/]
The idea is to create an  new list with its initial parent as int minimum.. -1 if input lists are all +ves

Iterative approach. While BOTH lists still have items. Check which needs to be added to the new list.
Add it and advance its pointer to the next item. Move the pointer of the new lists every time.

O(n+m), O(1)
``` 
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head=ListNode(-sys.maxsize)
        p = head
        while l1 and l2:
            if l1.val<l2.val:
                p.next=l1
                l1=l1.next
            else:
                p.next=l2
                l2 = l2.next
            p = p.next # p now points to last guy
        
        p.next = l1 if l1 is not None else l2
        return head.next
```

Recursive approach. Base case is when either lists is None, you just return the other.

The remaining logic is then
The smaller of the two lists' heads plus the result of a merge on the rest of the elements.

Specifically, if either of l1 or l2 is initially null, 
there is no merge to perform, so we simply return the non-null list. 
Otherwise, we determine which of l1 and l2 has a smaller head, 
and recursively set the next value for that head to the next merge result. 
Given that both lists are null-terminated, the recursion will eventually terminate.

O(n+m), O(n+m)
``` 
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

11. Intersection between two aarays. [https://leetcode.com/problems/intersection-of-two-arrays-ii/]
 
 Given 2 arrays return an array containing the intersection of the 2 arrays. 
 
 a = [1, 0, 0, 4, 2] <br>
 b= [4, 0, 0, 4] <br>
 result = [0, 0, 4] <br>
 
 First thing to do is sort the list. This should be one of the first things you consider with problems like this.
 
 Sorting will reveal some clues.<br>
 a = [0, 0, 1, 2, 4] <br>
 b = [0, 0, 4, 4] <br>
 result = [0, 0, 4] <br>
 Nice, so we can use 2 pointers to loop though both arrays checking in both items at the indexes are qual.
 We need to account for the difference in lengths.
 
 O(n) O(1)
 ``` 
 class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        res = []
        
        p1, p2 = 0 , 0
        while p1 < len(nums1) and p2 < len(nums2):
            if nums1[p1] == nums2[p2]:
                res.append(nums1[p1])
                p1+=1
                p2+=1
            else:
                if nums1[p1] < nums2[p2]:
                    p1+=1
                else:
                    p2+=1
        return res
 ```
 
 We could also use a hash map to store the frequencies of each number in liat 1.
 Then we iterate through the 2nd list and insert all the element that are in the 2nd list and have a count above 0
 in the map. We also decrement the count in the map for each match found. This prevents us from adddind duplicates 
 wrongly.
 
 O(n+m), O(n)
 ``` 
 class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        count_map = collections.Counter(nums1)
        res = []
        for i in nums2:
            if count_map.get(i, 0) > 0:
                res.append(i)
                count_map[i] = count_map[i]-1
        return res
 ```
 
 12. Best time to buy and sell stock. [https://leetcode.com/problems/best-time-to-buy-and-sell-stock/]
 
 The objective is to figure out the single best time to buy and single best time to sell to maximize profit.
 Just a single transaction. Buy and sell
 
 For these type of questions, try drawing the peaks and valleys.
 
 Brute force approach is to have a nested loop. For each price, check the remaining of the array for the max possible
  profit.
  
  O(n^2), O(1)
  
  ``` python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            profit = 0
            for i in range(len(prices)):
                for j in range(i+1, len(prices)):
                    profit = max(profit, prices[j]-prices[i])
        return profit
  ```
  
  The optimal solution is to keep track for the min price seen so far.
  As you progress down the array you check to see if the current price will yielf to more profit than seen 
  before. Remember to always keep track of the minimum price seen at the end of each iteration.
  
  O(n), O(1)
  
  ``` 
  class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        mnp = sys.maxsize        
        for i in range(len(prices)):
            profit = max(profit, prices[i] - mnp) # check
            if prices[i] < mnp: # update
                mnp = prices[i]
        return profit
  ```
  13. Excel sheet column number [https://leetcode.com/problems/excel-sheet-column-number/]
  
  A -> 1<br>
  B -> 2<br>
  C -> 3<br>
  Z -> 26<br>
  AA -> 27: 26+1 <br>
  AB -> 28 (26)+(2) <br>
  AZ -> 52: (26) + (26) <br>
  BA -> 53: (26*2)+(1) <br>
  CC -> 81: (26*3)+(3) <br>
  ZY -> 701<br>
  ZZ -> 702<br>
  AAA -> 703: (1*(26*26)) + (1*(26)) + (1) <br>
  AAB -> 704
  
  
  The approach here is to write out as many examples as possible to detect a pattern.
  The pattern here is simple.
  
  A-Z map to 1-26
  from left to right, the significance of the value increases
  For example AB
  A has a larger significance than B. Its Value is 26 and B is 2. 26+2 = 28
  The pattern here is that the more significant digits have a higher exponent for 26.
  the least exponent is 0. which is for single characters. A -> (1*26^0), B -> (2*26^0)
  
  So the component to figure out is the expoenent value.
  From right to left of the array, it starts at 0  and increases by one every time.
  
  O(n) O(1)
  ``` 
  class Solution:
    def titleToNumber(self, s: str) -> int:
        number = 0
        n = len(s)
        A = ord('A')
        exp = 0
        for i in range(n-1, -1, -1):
            char = ord(s[i]) - A + 1 # because a maps to 1 not 0
            number+= (char) * pow(26,exp)
            exp+=1
        return number
  ```
  14. Merge sorted arrays in place. [https://leetcode.com/problems/merge-sorted-array/submissions/]
  
  The optimal solution takes advatage of the fact that they sorted to get O(n+m) runtime.
  
  The fact that nums1 is large enough for both nums1 and nums2 data to fit in is major key.
  It means we can look at nums1[:m] as input1, and the nums1[:] as the result array.
  Now we can just fill nums1 with the data input1 and nums2 in sorted order.
  Makinf sure to handle over flows/ when one array still has data and the other is done...we just append whats left 
  to the result.
  
  Now this leads to na O(n+m) time and O(m) space complexity where m is the number of initialized items in nums1.
  We can do better. WITH 3 POINTERS! we can loop backwards, p1 = m-1, p2 = n-1 and p = m+n-1.
  p is used to fill the result array.<br>
  p1 is used to itertate nums1 (starting from the last intitialized index and walking back)<br>
  p2 is used to iterate nums2 (starting form the last initialized index)
  
  One of the 2 pointers will reach the end before the other. this is expected and gauranteed.
  If it is p2, then that's fine. It means all of nums2 items have been inserted into the correct locations in nums1.
  This also implies that whatever is still left in nums1 does not need to be moved, since nums1 is sorted.
  The items in p1 then we have a probelem. it means there are items in nums2 that are smaller than all the items in p1
  so they need to handled explicitly. They'd effectovely be inserted in fron of nums1.
  
  
  ``` 
  def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        
        p = m+n-1
        p1 = m-1
        p2 = n-1
        
        while p1>=0 and p2>=0:
            if nums1[p1]>nums2[p2]:
                nums1[p] = nums1[p1]
                p1-=1
            else:
                nums1[p] = nums2[p2]
                p2-=1
            p-=1
        
        while p2>=0: 
            nums1[p] = nums2[p2]
            p2-=1
            p-=1
            
  ```
  
  Since p2 is the real terminator of the while loops here, we can be smart and use on while loop.
  This means we will need to check explicitly for p1 in the loop before using it.
  ``` 
  def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        
        p = m+n-1
        p1 = m-1
        p2 = n-1
        
        while p2>=0:
            if p1 >=0 and nums1[p1]>nums2[p2]: # check that p1 is still safe!!
                nums1[p] = nums1[p1]
                p1-=1
            else: # just dump nums2 items in
                nums1[p] = nums2[p2]
                p2-=1
            p-=1
  ```
  
  
  
 
  15. Implement a min stack. [https://leetcode.com/problems/min-stack/]
  
  Requirement: O(1) time for push, pop, top and getMin
  
  There are 3 ways to attack this problem.
  1. Use to stacks. One for the actual data, the other 2 store the minimum of the items currently in the other stack.
  ``` 
 class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = collections.deque()
        self.minstack = collections.deque()
    def push(self, x: int) -> None:
        self.stack.appendleft(x)
        if not self.minstack:
            self.minstack.appendleft(x)
        else:
            if x<=self.minstack[0]:
                self.minstack.appendleft(x)
    def pop(self) -> None:
        item = self.stack.popleft()
        if self.minstack and item == self.minstack[0]:
            self.minstack.popleft()
            
    def top(self) -> int:
        if self.stack:
            return self.stack[0]

    def getMin(self) -> int:
        return self.minstack[0] if self.minstack else None
  ```
  ``` 
  Discovered this later.
  When using deque from collections.
  For queue behaviour, use append() and popleft()
  q = collections.deque()
  q.append(1)
  q.append(2)
  q.append(3)
  q.append(4)
  q.popleft() -> 1
  q.popleft() -> 2
  q.popleft() -> 3
  
  For stack behaviour really consider just using []
  if yous must, use append() and pop() like you will with []. same interface
  st = collections.deque()
  st = collections.deque()
  st.append(1)
  st.append(2)
  st.append(3)
  st.pop() -> 3
  st.pop() -> 2
  st.pop() -> 1
  ```
  
  2. Use an extra filed that maintains the current min. It gets updated and pushed into the stack whenever it is 
  no longer the minimum. The idea is that whenever the minimum item is popped out of the stack. We have to pop a 
  second time to obtain the new minimum which walsy follow its replcement. (rememebr what happend when a push smaller
   number is pushed in). Beatiful solution, but the lenght of the stack is warped!
   
   ```
    class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = collections.deque()
        self.min=sys.maxsize
    def push(self, x: int) -> None:
        if x<=self.min:
            self.stack.appendleft(self.min)
            self.min = x
        self.stack.appendleft(x)
    def pop(self) -> None:
        item = self.stack.popleft()
        if item==self.min and self.stack:
            self.min = self.stack.popleft()

    def top(self) -> int:
        if self.stack:
            return self.stack[0]

    def getMin(self) -> int:
        return self.min
   ```
   
   3. Store items in the stack as tuples or a data structure that always has a reference to the current min item in the 
   stack.
   So during push, this value is always set. It changes whenever a the item being pushed in is smaller than the 
   current min.
   
   ``` 
   class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = collections.deque()
    def push(self, x: int) -> None:
        if not self.stack:
            self.stack.appendleft((x, x))
        else:
            self.stack.appendleft((x, min(x, self.stack[0][1])))
    def pop(self) -> None:
        self.stack.popleft()
            
    def top(self) -> int:
        if self.stack:
            return self.stack[0][0]

    def getMin(self) -> int:
        return self.stack[0][1] if self.stack else None
   ```
  16. Remove duplicates from sorted array. [https://leetcode.com/problems/remove-duplicates-from-sorted-array/]
   
   2 Pointers baby. One to use as index for insertion, one to skip duplicates.
   
   
   O(n), O(1)
   
   ``` 
   def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        p1 = 0
        p2 = 0
        n = len(nums)
        while p1<n and p2<n:
            while p2<n and nums[p1] == nums[p2]:
                p2+=1
            if p2<n:
                p1+=1
                nums[p1] = nums[p2]
                
        return p1+1
   ```
   
   Another approach which shows that a simple for loop is fine. Less boundary check.
   ``` 
   def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        i = 0
        n = len(nums)
        for j in range(n):
            if nums[j] != nums[i]:
                i+=1
                nums[i] = nums[j]
        return i + 1
   ```
   
  
  17. Valid palindrome. [https://leetcode.com/problems/valid-palindrome/]
  
  Normaly, we can just use the 2 pointer technique and loop through from both ends while p1 > p2 checking if 
  characters are the same. This one is a little different. We need to ignore non alpha numeric characters!!!
   
   Consider madam: This is a palindrome.<br>
   madam; normally will not be a palindrome, but in this case it is.
   
   Now the challenge is determining characters that are alphanumeric.
   These are A-Z, a-z 1-9 and I think a few other weird ones.
   Anyways, python has a cool funciton for determining this. If not, we would have had to do some ord(char) stuff and
   check that it is within a range! probably, 65-b to something since A is 65 and a is 97, Also we ignore case, which
    is trivial really, just do comparisons on the lowercases
    
    
   O(n), O(1)
   ```
   class Solution:
    def isPalindrome(self, s: str) -> bool:
        if not s:
            return True
        p1 = 0
        p2 = len(s)-1
        while p1<p2:
            if not s[p1].isalnum():
                p1+=1
                continue
            if not s[p2].isalnum():
                p2 -=1
                continue
            if s[p1].lower() != s[p2].lower():
                return False
            p1 +=1 
            p2 -=1
            
        return True
   ```
  
  18. Longest common prefix. [https://leetcode.com/problems/longest-common-prefix/]
  
  Many ways to go about this problem, but in the end its actually very simple and brute force sorta is the most optimal.
  Well not weyrey brute force, but the solution that jumps at your first is pretty good.
  Get the length of the smallest string in the list. 
  This is the upper bound on the length of the longest substring!
  
  Approach 1: Divide and conquer. Realise that for arr [s1, s2, s3, s3] the lcp(arr) = LCP[lcp(arr[0:1]) and lcp
  (arr[1:4])]. lcp of left half and lcp of right half is the lcp of left half + right half.
  
  O(S) where S is the sum of all the characters in the strings. O(n) space. recursive calls. could be log(n) because 
  we finish the lefts stack before we start the rights stack.
  
  ``` 
  class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs)==0:
            return "" # obvious base case
        if len(strs) == 1:
            return strs[0] # critical base case
        
        n = len(strs)
        mid = n//2 # we use // to get mid n items
        
        left = self.longestCommonPrefix(strs[0:mid])
        right = self.longestCommonPrefix(strs[mid:n])
        return self.getLongestPrefix(left, right)
            
        
    def getLongestPrefix(self, str1, str2):
        n = min(len(str1), len(str2))
        for i in range(n):
            if str1[i] != str2[i]:
                return str1[0:i]
        return str1[0:n]
  ```
 
 Approach 2: Preferred!
 
 As usuall we get the upper bound of the longest common prefix which is lenght of the smallest string in the list.
 
 The idea is to start from index 0 and compare every character in the smallest string to the character at the same 
 index in the the other strings. Once we encounter a a character that is different at the same index in any of the 
 other strings, we terminate.
 
 The index where this happens is the length of the longest common prefix. We essentially
 assume the lcp is the smallest str and then ve validate each character from index zero terminating once we encounter
  a non matching character for the same index in any of the other strings. if we go all the way to the end of the
  smallest string, we return it as the lcp
 ``` 
 class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        smallest = ""
        n = sys.maxsize
        
        for i in range(len(strs)):
            if len(strs[i])<n:
                smallest = strs[i]
                n = len(smallest)
   
        for i in range(n)):
            for other in strs:
                if other[i] !=smallest[i]:
                    return smallest[:i] # everything from the beingin up to i(exclusive) is the lcp    
                
        return smallest
 ```
 19. Remove Nth Node form the end of a linked list [https://leetcode.com/problems/remove-nth-node-from-end-of-list/]
 
 Return its head after removing the nth node from the end.
 Dummy head is key to solving these they o problems!
 
 ``` 
 class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if not n :
            return head
        dummy = ListNode(0)
        dummy.next = head
        p1 = head
        
        i = 0
        while p1:
            p1 = p1.next
            i+=1

        i-=n
        p2 = dummy # use dummy when you need to relink/skip
        while i:
            p2 = p2.next
            i-=1
            
        p2.next = p2.next.next
        return dummy.next
 
 ```
 
 20. Remove duplicates from sorted linkedlist [https://leetcode.com/problems/remove-duplicates-from-sorted-list/]
 You really just need to iterate to one item before the end. For each node, compare its value to that of the node 
 after it.
 OO(n) O(1)
 
 ``` 
 class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        curr = head
        while curr and curr.next:
            if curr.val == curr.next.val:
                curr.next = curr.next.next
            else:
                curr = curr.next
        return head
 ```
 Another approach
 ``` 
 def deleteDuplicates(self, head):
    cur = head
    while cur:
        while cur.next and cur.next.val == cur.val:
            cur.next = cur.next.next     # skip duplicated node
        cur = cur.next     # not duplicate of current node, move to next node
    return head
 ```
 
 21. Linked list cycle. Detect a cycle. [https://leetcode.com/problems/linked-list-cycle/]

 The fats guy, slow guy approach
 ``` 
 class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head:
            return False
        
        fast_guy = head
        slow_guy = head
        
        while fast_guy and slow_guy:
            if not fast_guy.next:
                return False
            fast_guy = fast_guy.next.next
            slow_guy = slow_guy.next
            if fast_guy == slow_guy:
                return True
        return False
 ```
 
22.Remove element from linked list wit value val [https://leetcode.com/problems/remove-linked-list-elements/]

Whenevr you need to delete an item form a linkedlist and its possibe you have to delete the head, use a dummy head

``` 
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        p1 = head
        p2 = dummy
        
        while p1:
            if p1.val == val:
                p2.next = p2.next.next
                p1 = p2.next
            else:
                p1 = p1.next
                p2= p2.next
                
        return dummy.next
```

23. Middle element in linked list. [https://leetcode.com/problems/middle-of-the-linked-list/] 
You just need to find the length and then n//2 is the mid element.
If the length of the list is even, we will end up with m being the 2nd of the 2 middle elements
n = 4, n//2 = 2  [1, 2, 3, 4] which is element mid elements are 2 and 3. if we were to return the first, then we jsut
 need to reduce m by one if n is even.
 
 O(n) O(1)
 ``` 
 class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        n = 0
        p = head
        
        while p:
            p = p.next
            n+=1
        m = n//2
        p = head
        while m:
            print(m)
            p = p.next
            m-=1
        return p
 ```

We can also use a fast and slow pointer. Slow pointer will be at the mid or 2nd mid when fast is at the end.

O(n) O(1)
``` 
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        return slow
```


24. Plus one linked list. [https://leetcode.com/problems/plus-one-linked-list/]

Just like the array question, you just need to add one to the number. With the array we are able to loop backwards 
starting from the last number and mantaining a carry which is set to 1 when the current sum is 10. so carry=1 sum=9.

To be able to loop backwards, we can just revers the linked list, add 1 to the head, handle the carry if any do the sum 
and then unreverse it at the end.
We create and extra node with value 1 and put it at the tail if there is still a carry after adding just 1. 999->0001
 then we reverse it to get 1000

4->2 => 2->4 => add 1 => 3->4 => 35
9->9 => 9-9 => add 1 => 0->0->1 => 100

O(n), O(1)

``` 
class Solution:
    def plusOne(self, head: ListNode) -> ListNode:
        curr  = head
        def reverse(node):
            prev = None
            while node:
                nxt = node.next
                node.next = prev
                prev=node
                node=nxt
            return prev
        
        r_head = reverse(curr)
        
        r_head.val+=1
        if r_head.val==10:
            r_head.val = 0
            carry=1
            curr = r_head.next
            while curr and carry:
                curr.val+=carry
                if curr.val==10:
                    curr.val = 0
                    carry=1
                else:
                    carry=0
                curr = curr.next
            if carry: # still a carry and no curr, so end of list
                curr = r_head
                while curr.next: # because we need to guy before the end
                    curr = curr.next
                curr.next = ListNode(1)
            
        return reverse(r_head)
```

25 Copy list with random Pointer. [https://leetcode.com/problems/copy-list-with-random-pointer/]
Deep copy means a brand new object that just contains the same values for primitive types.
All references to objects are also recreated to point to depp copies

Great thing here is that Node objects can be used as keys for the hashmap
``` 
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None
        
        copy_by_org = {}
        curr = head
        while curr:
            copy_by_org[curr] = Node(curr.val, None, None)
            curr = curr.next
        curr = head
        
        while curr:
            new_node = copy_by_org[curr]
            nxt = curr.next
            rand = curr.random
            if nxt:
                new_node.next = copy_by_org[nxt]
            if rand:
                new_node.random = copy_by_org[rand]
            curr = curr.next
        return copy_by_org[head]
```

26. Clone Graph. [https://leetcode.com/problems/clone-graph/]

    Same logic to the above, I even learnt a better approach to reconstruction. We can just iterate over the key 
    value pairs in the hashmap. To be honest this question was very straightforward. Especially after solving the 
    linkedlist one. Its quite simple once you know how to keep track of items already seen!
    
    O(n), O(n)
    
   ``` 
   class Solution:
        def cloneGraph(self, node: 'Node') -> 'Node':
            if not node:
                return None
            
            st = [node]
            d = {}
            seen = set([])
            while st:
                old = st.pop()
                if old not in seen:
                    seen.add(old)
                    copy = Node(old.val, [])
                    d[old] = copy
                    for n in old.neighbors:
                        st.append(n) # this is key, so we can visit them later. stack is used, so dfs 
            
            for old , new in d.items():
                for n in old.neighbors:
                    new.neighbors.append(d[n])
            return d[node]
   ```
  27. Add two numbers presented as 2 linkedlists. [https://leetcode.com/problems/add-two-numbers-ii/]
  
  One approach is to reverse and add like you would a normal array. Handing all possible None exceptions as you 
  traverse the lists. Reversing makes it a lot easier because this is really how we appraoch suming numbers on paper
  
  O(n), O(n)
  
  ``` 
  class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        def reverse(node):
            prev = None
            while node:
                front = node.next
                node.next = prev
                prev = node
                node = front
            return prev
        n1 = reverse(l1)
        n2 = reverse(l2)
        carry = 0
        sm = n1.val+n2.val
        if sm >9:
            sm = sm%10 
            carry=1
        ans = ListNode(sm)
        ps = ans
        c1, c2 = n1.next, n2.next
        while c1 or c2:
            sm = carry
            if c1:
                sm +=c1.val
                c1=c1.next
            if c2:
                sm+=c2.val
                c2=c2.next
            if sm>9:
                sm = sm%10
                carry=1
            else:
                carry=0
            ps.next = ListNode(sm)
            ps = ps.next
        if carry:
            ps.next = ListNode(1)
        return reverse(ans)
  ```
  
  28. Partition list. [https://leetcode.com/problems/partition-list/]
  
  Given in a linked list and a value x, partition the linked list such that nodes less than x come before nodes that 
  are greater than or equal to x. Preserve the original order of the the nodes.
  
  We just need to create 2 lists. one for the items less than x the other for the items greater than x. At the end we
   merge them. We need dummy heads for the new lists. We need to have access to the next without overriding values in 
   the original list. we just need to re point the nodes nexts. Its 10 times easier with the dummy nodes
   
   O(n), O(1)
   ``` 
   class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        h1 = l1 = ListNode(0)
        h2 = l2 = ListNode(0)
        p = head
        while p:
            if p.val<x:
                l1.next = p
                l1=l1.next
            else:
                l2.next = p
                l2=l2.next
            p = p.next
        l2.next=None # the end of the list or else itd point to whatever p.next pointed to
        l1.next=h2.next # connect both lists
        return h1.next # skip the dummy head
   ```
  
  29. Swap Nodes in Pairs [https://leetcode.com/problems/swap-nodes-in-pairs/]
    Given a linked list, swap every two adjacent nodes and return the head
    
   1-2-3 => 2->1->3
   1->2->3->4 2->1->4->3
    
   Notice the swapping is done in pairs! I.e only when we 2 complete nodes.
   so base case is return head when we dont have up to 2 nodes in the list.
   Questions like this are best easily solved by considering easier solutions and building from that.
   Consider a list with 1 , 2, then 3 items. You'd see the recursion on the 3rd
   
  O(n), O(n/2)
   ```
   class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head:
            return head
        if not head.next:
            return head

        front = head.next # we need a reference to this guys so we can return it as the head later :)
        head.next= self.swapPairs(head.next.next)
        front.next = head # the actual swapping 
        return front
    
   ```
   Another approach is to do it iteratively using s dummy head as the curr and working on the curr.next and curr.next
   .next
   
   ``` 
   class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        prev, cur = dummy, head
        
        while cur and cur.next:
            front = cur.next.next
            prev.next = cur.next
            
            cur.next.next = cur
            cur.next = front
            
            prev=cur 
            cur=front
        return dummy.next
   ```
        
  30. Flowers planting with no adjacent. Graph question. [https://leetcode.com/problems/flower-planting-with-no-adjacent/]
  
  This question is very simple once you understand what it is asking for.
  You are give N gardens and a grid that represents the paths between the gardens. paths = [[X, y],[y, a], [b, c]]
  This means garden x and y are connected, y and a are connected, b and c are connected. 
  
  From this information we can build an adjency list using a graph. From the adjanecy list we can make sure not to 
  plant a flower in a garden if any of its neighbours already have that flower. There are only 4 types of flowers
   
   
  O(n), O(n)
  
  ``` 
  class Solution:
    def gardenNoAdj(self, N: int, paths: List[List[int]]) -> List[int]:
        res = [1 for i in range(N)]
        adj_list = collections.defaultdict(set)
        
        for x, y in paths:
            adj_list[x].add(y)
            adj_list[y].add(x)
        
        for grd, neis in adj_list.items():
            used = set()
            
            for n in neis:
                used.add(res[n-1])
            for f in range(1, 5):
                if f not in used:
                    res[grd-1]=f
                    break
                    
        return res
  ```
  
  31. Find the Town judge. [https://leetcode.com/problems/find-the-town-judge/]
  Just need to ensure that after building the graph, only one person is not a key. This is guy is likely to be the 
  judge. we then need to verify that he is trusted by all. If there is more than one person without a key in the 
  graph, then we return -1 because more than one person does not trust anyone which means there is no judge.
  
  O(n), O(n)
  
  ``` 
  class Solution:
    def findJudge(self, N: int, trust: List[List[int]]) -> int:
        
        adj = collections.defaultdict(set)
        
        for a, b in trust:
            adj[a].add(b)
        judges = set([])
        for j in range(1, N+1):
            if j not in adj.keys():
                judges.add(j)
        if len(judges)!=1:
            return -1
        j  = judges.pop()
        for k, v in adj.items():
            if j not in v:
                return -1
        return j
  ```
  32. Reconstruct itinerary. [https://leetcode.com/problems/reconstruct-itinerary/]
  
  Depth first search after sorting the edges in the graph in reverse lexical order.
  Return the result reversed, because wev'e appended the first last based on dfs using a stack.
  
  
  ``` 
  class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        graph = collections.defaultdict(list)
        for c, d in tickets:
            graph[c].append(d)
        for k in graph.keys():
            graph[k].sort(reverse=True)
            print(k, graph[k])

        res = []
        key = "JFK"
        stack = [key]
        while stack:
            key=stack[-1]
            # print(stack)
            while(graph[key]):
                print("loop",stack)
                key = graph[key].pop()
                stack.append(key)
            res.append(stack.pop())
        return res[::-1]
  ```
  
  33. Binary Tree paths. [https://leetcode.com/problems/binary-tree-paths/]
  
  We want to get all the paths that lead from head to a leaf node. We can use dfs recursively and iteratively.
  
  The main trick here is to send a copy of the paths so far not a reference, so that each node has its own paths and 
  you are'nt updating the global paths for everyone.
  
  
  ``` 
  class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if not root:
            return []
        paths = []
        head= root
        stack = [(head, [])]
        while stack:
            node, path = stack.pop()
            path.append(str(node.val))
            if not node.left and not node.right:
                paths.append(path)
            if node.left:
                stack.append((node.left, path.copy()))
            if node.right:
                stack.append((node.right, path.copy()))
        for i in range(len(paths)):
            paths[i] = "->".join(paths[i])
        return paths
  ```
  Same trick, pass down the copyof the list of the nodes parents
  ``` 
  class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        
        def dfs(node, path, paths):
            if node:
                path.append(str(node.val))
                if not node.left and not node.right:
                    paths.append(path)
                else:
                    dfs(node.left, path.copy(), paths)
                    dfs(node.right, path.copy(), paths)
        a = []
        b = []
        dfs(root, a, b)
        for i in range(len(b)):
            b[i]="->".join(b[i])
            
        return b
  ```
  
  34. Balance Binary Tree. [https://leetcode.com/problems/balanced-binary-tree/]
   
   The concept is quite interesting. It depends on the ability to compute the height of a tree giving the node and 
   using dfs.
   
   Approach 1: Will calculate the height of the left sub tree and right subtree of every node, 
   starting from the head. If at any point the Balanced tree variant fails, 
   ie abs(left-right)>1 we return False. So the result is an and between the recursive result on 
   the left and right and the terminating case is the variants failure.
   
   ``` 
   class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
    
        def geth(node):
            if not node:
                return 0
            left = geth(node.left)+1
            right = geth(node.right)+1
            return max(left, right) # only one of the 1s added makes it through to the top
        
        if not root: # base case
            return True
        
        if abs(geth(root.left)-geth(root.right))>1: # balanced tree defintion. False if fails
            return False
        
        return self.isBalanced(root.left) and self.isBalanced(root.right) 
        # check sub trees because they need to satisfy independently
   ```
    
   Approach 2 avoids lots of repeated calculations done by approach 1. We calculate the high of the leaf nodes and 
   verify that they balanced before checking parent nodes.
   
   ```
   class Solution:
    def isBalancedHelper(self, root):
        if not root: # an empty tree is balanced
            return True, 0
        # check the subtrees to see if they are balanced
        l_balanced, lh = self.isBalancedHelper(root.left) # we go deeper and deeper
        if not l_balanced:
            return False, 0 # does not matter the heigt
        r_balanced, rh = self.isBalancedHelper(root.right) # we go right, then left depeer and deeper
        if not r_balanced: # quick terminate
            return False, 0
        rh+=1
        lh+=1
        return abs(lh-rh)<=1, max(lh, rh) # this is not where recursion happens
    
    def isBalanced(self, root: TreeNode) -> bool:
        return self.isBalancedHelper(root)[0]
   ```
   
   
35. Minimum depth of binary tree. [https://leetcode.com/problems/minimum-depth-of-binary-tree/]
At first this problem seems easy. lol but it is not. You cant just calculate the height of the left sub tree and the 
right subtree and return the minimum. That will be wrong in some case.

   ```
    3
   / \
  9  20
    /  \
   15   7
 The anser is 2. 3->9 so 2 nodes. this case will work if we just retured the min of the left and right heights of the
  root node.
  
     3
   / 
  9 
 The minum of the 2 heights here is 0 so we return 0+1 but the anser is 2. It has to be a leaf node.
 
 
This solution using a stack and dfs accounts for that.

O(n) and O(n)
class Solution:

    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        
        stack = [(root, 1)]
        mind = sys.maxsize
        while stack:
            node, depth = stack.pop()
            if not node.left and not node.right:
                mind = min(depth, mind)
            
            depth+=1
            if node.left:
                stack.append((node.left, depth))
            if node.right:
                stack.append((node.right, depth))
        return mind
 ``` 
 We can also use bfs. We use a que instead. This guys is more optimal because it terminates at the level with the 
 shortest path and does not traverse the whole tree.
 ``` python 
 so with tree 
    3
   / \
  9  20
    /  \
   15   7
   
 We will only visit node 3, 2, and 20. Once we enounter a leaf we terminated. 
 level order makes the most sense for this question.
 
 from collections import deque
class Solution:

    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        q = deque()
        depth = 1
        q.append((root, depth))
        
        while q:
            node, depth = q.popleft()
            if not node.left and not node.right:
                return depth
            depth+=1
            if node.left:
                q.append((node.left, depth))
            if node.right:
                q.append((node.right, depth))
        return depth
 ```
 
 36. Binary tree path sum. [https://leetcode.com/problems/path-sum/solution/]
 
 Given a binary tree and a sum, determine if the tree has a root-to-leaf path such 
 that adding up all the values along the path equals the given sum.
 
 This is a pretty straight forward question. My first approach worked. use a que or stack and store the current sum 
 together with the node in the stack or que
 
 ```
 from collections import deque
class Solution:
    def hasPathSum(self, root: TreeNode, sm: int) -> bool:
        if not root:
            return False
        if not root.left and not root.right:
            return root.val == sm
        
        q = deque()
        q.append((root, 0))
        while q:
            node, tsm = q.popleft()
            tsm+=node.val
            if not node.left and not node.right:
                if tsm == sm:
                    return True
            if node.left:
                q.append((node.left, tsm))
            if node.right:
                q.append((node.right, tsm))
        return False
        
 
  we can also use use dfs i.e a stack 
  
  class Solution:
    def hasPathSum(self, root: TreeNode, sm: int) -> bool:
        if not root:
            return False

        if not root.left and not root.right:
            return root.val == sm
        st = [(root, 0)]
        
        while st:
            node, tsm = st.pop()
            tsm+=node.val
            if not node.left and not node.right:
                if tsm == sm:
                    return True
            if node.left:
                st.append((node.left, tsm))
            if node.right:
                st.append((node.right, tsm))
        
        return False
 ```
 
 Another interesting approach is to use recursion and keep decreasing the sum value as we recur down.
 
 ``` 
from collections import deque
class Solution:
    def hasPathSum(self, root: TreeNode, sm: int) -> bool:
        if not root:
            return False

        if not root.left and not root.right:
            return root.val == sm
        sm-=root.val
        
        return self.hasPathSum(root.left, sm) or self.hasPathSum(root.right, sm)
 ```

37. Path Sum III. [https://leetcode.com/problems/path-sum-iii/]


Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, 
but it must go downwards (traveling only from parent nodes to child nodes).

O(n) O(n*n)
```
class Solution:

    def pathSum(self, root: TreeNode, s: int) -> int:
    
        def check(node, target, n):
            if not node:
                return
            if node.val == target:
                n[0]+=1
                # dont return because we can still get nodes that sum up to the target -ves and +vs
            if node.left:
                check(node.left, target-node.val, n)
            if node.right:
                check(node.right, target-node.val, n)
        
        def dfs(node, target, n):
            if not node:
                return
            check(node, target, n)
            if node.left:
                dfs(node.left, target, n)
            if node.right:
                dfs(node.right, target, n)
            
        n = [0]
        dfs(root, s, n)
        return n[0]
```
Another Approach. [https://leetcode.com/problems/path-sum-iii/]

We maintain a dictionary of past sub trees that if remoevd from the current subtree could lead to it being a sub path
 that sums up to target K. Look at quesiton 38 below if you do not understand the approach. It is similar to sub 
 array sums
 
 ``` 
class Solution:
    n = 0 # class variable bu accessed through self
    def pathSum(self, root: TreeNode, s: int) -> int:
    
        
        def dfs(node, target, cum, mp):
            if not node:
                return
            cum+= node.val
            x = cum-target
            if x in mp:
                self.n+=mp[x]
            mp[cum] = mp.get(cum, 0)+1
            if node.left:
                dfs(node.left, target, cum, mp)
            if node.right:
                dfs(node.right, target, cum, mp)
            
            mp[cum]-=1
            
        dfs(root, s, 0, {0:1})
        return self.n
 ```
 
 38. Subarray equals k. [https://leetcode.com/problems/subarray-sum-equals-k/]
 
 We need to count the number of sub arrays whose sume equals k. Quite nice question. You need to think deeply to see 
 that memoization helps a lot.
 
```
consider array [4, 0, 3, 0, 4, 3, 7, 0] and  k = 7

[4, 0, 3] =>7
[4, 0, 3, 0] =>7
[0, 3, 0, 4] =>7
[3, 0, 4] =>7
[0, 4, 3] =>7
[4, 3] =>7
[7] =>7
[7, 0] =>7
8 sub arrays
 
Notice that zeroes and also negatives have a weird impact on the size and number fo sub arrays.



The optimal approach is to keep track of the count of all sums see as we travers the array
{0:1} starting with 0 because it is always available as a prefix. meaning no other items is to be used to make up the
 sum at the current element if the current element =k meh that may be difficult to understand.
 
 
 
 
Alright. At every index, we have the cumulative sum. We will like to know if there is an x such that 
cum_sum-x = k. That is there is a sub array prior to this item whose sum is x and if removed from the current 
cumulative summ will lead to the target.

at index 2 arr[3] =0
There is a subarray whose sum when subtracted from the current summlative subarray will lead to 7
 i.e sub array []. if you remove this sub array from sub array [4, 0, 3, 0] you end hav a subaarray whose sum is the 
 target.
 Another case. at index 4 arr[4] = 4 sub array [4, 0, 3, 0, 4]. There is another subarray if removed from the sub 
 array willresult in a subarray whose sum is 7. That subarry is [4]. hmmm there are 2 such subarrays, [4] and [4, 0] 
 leaving us with [0, 3, 0, 4] and [3, 0, 4] respectively.
 
 
 so at everypoint, we need to update the map with the count of the current cumulative sum, so if any subarrays need 
 that difference taken out to give them the target, they can use it the number of times it has appeared.
 curr_sum -x = target. if x is in the map then yippy, we can use x map[x] times, we increment count this number of times
 
 
 class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        count = 0
        # sums = [0 for _ in range(len(nums))]
        map = collections.defaultdict(int)
        map[0]=1
        sm = 0
        for i in range(len(nums)):
            sm+=nums[i]
            if sm-k  in map:
                count+=map[sm-k]
            map[sm]+=1
            print(map)
            print(i, count)
        return count
```


39. Path sum iii. [https://leetcode.com/problems/path-sum-iii/]

Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

Remember to pass a copy of the dame paths

O(n) O(n)
``` 
class Solution:
    def pathSum(self, root: TreeNode, m: int) -> List[List[int]]:
        
        if not root:
            return []
        
        st = [(root, 0, [])]
        paths =[]
        while st:
            node, sm, path = st.pop()
            curr_sum = node.val+sm
            path.append(node.val)
            
            if not node.left and not node.right and curr_sum == m:
                paths.append(path)
            
            if node.left:
                st.append((node.left, curr_sum, path.copy()))
            if node.right:
                st.append((node.right, curr_sum, path.copy()))
        return paths
```
40. Nested List weight Sum. [https://leetcode.com/problems/nested-list-weight-sum/]

 Given a nested list of integers, return the sum of all integers in the list weighted by their depth.

Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Input: [1,[4,[6]]] <br>
Output: 27 <br>
Explanation: One 1 at depth 1, one 4 at depth 2, and one 6 at depth 3; 1 + 4*2 + 6*3 = 27.

We can use dfs iteratvely and recursively.
``` 
class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        
        if not nestedList:
            return 0
        
        st = [(n,1) for n in nestedList]
        sm = 0
        while st:
            n, d = st.pop()
            if n.isInteger():
                sm+=d*n.getInteger()
            else:
                d+=1
                for l in n.getList():
                    st.append((l, d))
        return sm
```
 
 Recursively
 ``` 
 class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        if not nestedList:
            return 0
        
        def dfs(nl, depth):
            if not nl:
                return 0
            sm = 0
            for n in nl:
                if n.isInteger():
                    sm+=n.getInteger()*depth
                else:
                    sm+=dfs(n.getList(), depth+1)
            return sm
        return dfs(nestedList, 1)
 ```
 
41. Decode String. [https://leetcode.com/problems/decode-string/]

You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

``` 
s = "3[a]2[bc]", return "aaabcbc".
s = "3[a2[c]]", return "accaccacc".
s = "2[abc]3[cd]ef", return "abcabccdcdcdef".
```
We just iterate through every character and evelauate every inner []. To do this we look for ] and then pop until we 
see a [
``` 
class Solution:
    def decodeString(self, s: str) -> str:
        st = []
        
        for c  in s:
            if c =="]":
                word = ""
                while st:
                    w = st.pop()
                    if w == "[":
                        break
                    word = w+word
                num = ""
                while st:
                    d = st.pop()
                    if d.isdigit():
                        num = d+num
                    else:
                        st.append(d)
                        break
                if num:
                    print(num)
                    word = int(num)*word
                st.append(word)
            else:
                st.append(c)
        return "".join(st)
```


42. Max Area of Island. [https://leetcode.com/problems/max-area-of-island/]

Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

Find the maximum area of an island in the given 2D array. (If there is no island, the maximum area is 0.)


Really simple when you handle all boundaries and a bit of recursion.

``` 
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        gmax = 0
        if not grid:
            return gmax
        
        rows = len(grid)
        cols = len(grid[0])    
        
        def getArea(i, j):
            count=0
            r = i+1
            while r<rows: # go down
                if grid[r][j]==1:
                    count+=1
                    grid[r][j]=None
                    count+=getArea(r,j)
                else:
                    break
                r+=1
            r=i-1
            while r>=0:
                if grid[r][j]==1:
                    count+=1
                    grid[r][j]=None
                    count+=getArea(r,j)
                else:
                    break
                r-=1
            c = j+1
            while c<cols:
                if grid[i][c]==1:
                    count+=1
                    grid[i][c]=None
                    count+=getArea(i,c)
                else:
                    break
                c+=1
            c = j-1
            while c>=0:
                if grid[i][c]==1:
                    count+=1
                    grid[i][c]=None
                    count+=getArea(i,c)
                else:
                    break
                c-=1
            return count
            
        for i in range(rows):
            for j in range(cols):
                if grid[i][j]==1:
                    grid[i][j] = None
                    count= getArea(i, j)+1
                    gmax = max(count, gmax)
        return gmax
```

43. Convert Sorted List to Binary Search Tree. [https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/]

Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

The middle item is the root node. That's all you need to solve this problem.

n <=3 are base cases that can be handled real quick. n=4 too.
``` 
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        if not head:
            return None
        def getMid(node):
            slow = fast = node
            p=node
            size = 0
            while p:
                p = p.next
                size+=1
            prev = slow 
            while fast and fast.next:
                prev = slow
                slow = slow.next
                fast = fast.next.next
            prev.next = None # disconnect 
            return slow, size
        
        def bst(node, n=None):
            if n==3:
                top = TreeNode(node.next.val)
                top.left = TreeNode(node.val)
                top.right = TreeNode(node.next.next.val)
                return top
            if n==2:
                top = TreeNode(node.next.val)
                top.left = TreeNode(node.val)
                return top
            if n==1:
                return TreeNode(node.val)
            if n==0:
                return None
            mid, n = getMid(node)
            if n%2:
                mid.right = bst(mid.next, n//2)
                mid.left = bst(node, n//2)
            else:
                mid.right = bst(mid.next, (n//2)-1)
                mid.left = bst(node, n//2)
            return mid
        return bst(head)
```

44. Binary Tree Level Order Traversal II

Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to root).
BFS with control on the level.

``` 
from collections import deque
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        paths = []
        
        q = deque()
        q.append(root)
        
        while q:
            size = len(q)
            level = []
            for i in range(size):
                node = q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            paths.append(level)

        paths.reverse()
        return paths
```

45. Employee importance. https://leetcode.com/problems/employee-importance/

Now given the employee information of a company, and an employee id, you need to return the total importance value of this employee and all his subordinates.

``` 
"""
# Employee info
class Employee:
    def __init__(self, id, importance, subordinates):
        # It's the unique id of each node.
        # unique id of this employee
        self.id = id
        # the importance value of this employee
        self.importance = importance
        # the id of direct subordinates
        self.subordinates = subordinates
"""
from collections import deque
class Solution:
    def getImportance(self, employees, id):
        """
        :type employees: Employee
        :type id: int
        :rtype: int
        """
        
        emp_by_id = {}
        empl = None
        for emp in employees:
            if emp.id == id:
                empl = emp
                continue
            emp_by_id[emp.id] = emp
        
        total = empl.importance
        q = deque(empl.subordinates)
        
        while q:
            e = q.popleft()
            emp = emp_by_id[e]
            subs = emp.subordinates
            for i in subs:
                q.append(i)
            total+=emp.importance
        return total
        
 # recursive bfs
class Solution(object):
    def getImportance(self, employees, query_id):
        emap = {e.id: e for e in employees}
        def dfs(eid):
            employee = emap[eid]
            return (employee.importance +
                    sum(dfs(eid) for eid in employee.subordinates))
        return dfs(query_id)
```
46. Maximum Depth of N-ary Tree

Given a n-ary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Pretty standard, we can solve it iteratively or recursively. The main deal is that we loop over the children of each 
node as we do not know how many they are. left, right will not cut it

``` 
from collections import deque
class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        
        q = deque()
        q.append((root, 1))
        max_d = -1
        while q:
            node, d = q.popleft()
            if node.children:
                for ch in node.children:
                    q.append((ch, d+1))
            else:
                max_d = max(d, max_d)
        return max_d
```

``` 
class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        max_d=1
        for ch in root.children:
            h = self.maxDepth(ch)+1
            max_d = max(h, max_d)
            
        return max_d
```

47. Cousins in binary tree. https://leetcode.com/problems/cousins-in-binary-tree/

Two nodes of a binary tree are cousins if they have the same depth, but have different parents.
Return true if and only if the nodes corresponding to the values x and y are cousins.

``` 
from collections import deque
class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        if not root:
            return False
        q = deque()
        q.append((root, None))
        while q:
            n = len(q)
            parent_x = parent_y = None
            for _ in range(n):
                node, parent= q.popleft()
                if node.left:
                    q.append((node.left, node))
                if node.right:
                    q.append((node.right, node))
                if node.val == x:
                    parent_x = parent
                if node.val == y:
                    parent_y = parent
                if parent_x and parent_y:
                    if parent_x != parent_y:
                        return True
                    else:
                        return False
        return False
```
48. Binary Tree Right Side View. https://leetcode.com/problems/binary-tree-right-side-view/

Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

Loved this question. The trick is to visit the right most node of each level first. Then for each level, add this 
right most node to the result array. BFS to the rescue with an inner loop to separate levels.

``` 
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        q = deque()
        q.append(root)
        right = []
        while q:
            node = q.popleft()
            n = len(q) # after weve removed the righ most guy.. so the rest
            right.append(node.val)
            if node.right:
                q.append(node.right)
            if node.left:
                q.append(node.left)
            for _ in range(n): # just this level
                node = q.popleft()
                if node.right:
                    q.append(node.right)
                if node.left:
                    q.append(node.left)
        return right
```

49. 3 SUM. https://leetcode.com/problems/3sum/submissions/
Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note:
The solution set must not contain duplicate triplets.

``` 
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ans = []
        n = len(nums)
        nums.sort()
        for i in range(n-2):
            if i and nums[i] == nums[i-1]:
                continue
            else:
                j = i+1
                k = n-1
                while j<k:
                    print(i, j, k)
                    if k<n-1 and nums[k] == nums[k+1]:
                        k-=1
                        continue
                    if nums[i]+nums[j]+nums[k]>0:
                        k-=1
                        continue
                    elif nums[i]+nums[j]+nums[k]<0:
                        j+=1
                        continue
                    else:
                        ans.append([nums[i],nums[j], nums[k]])
                        k-=1
                        j+=1
        return ans
```

50. Product of array except self. https://leetcode.com/problems/product-of-array-except-self/

Given an array nums of n integers where n > 1,
return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

It seems simple but it is not when you consider inputs like [1,0] or [0,0, 1, 2]

It gets even more complex when you arent allowed to use the division operation.

```
class Solution:
    
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if not nums:
            return [] 
        left = [1 for _ in nums]
        right =[1 for _ in nums]
        ans = [0 for _ in nums]
        n = len(nums)
        for i in range(1, n): # start from 2nd element
            left[i] = nums[i-1]*left[i-1]
        
        for i in range(-2, -(n+1), -1): # start from 2nd to last
            right[i] = nums[i+1] * right[i+1]
       
        for i in range(n):
            ans[i] = right[i] * left[i]        
        return ans
```
Another approach with less space. R or L can be computed on the fly..
Weve done R but L might actually be easier. no need for the reverse iteration.

-1, -(n+1), -1 => -1 all the way to the first element.
``` 
class Solution:
    
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        
        left = [1 for _ in nums]
        r = 1
        ans = [0 for _ in nums]
        n = len(nums)
        for i in range(1, n): # start from 2nd element
            left[i] = nums[i-1]*left[i-1]
        
        for i in range(-1, -(n+1), -1): # -1 to -(n+1) so all items!
            ans[i] = r * left[i]
            r = r*nums[i]
            
        return ans
```

51. Maximum product array. https://leetcode.com/problems/maximum-product-subarray/

Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product

Think about negative numbers and their impact on sub array products. also remember its contiguos subarray.
``` 
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        imax = imin = mx = nums[0]
        n = len(nums)
        for i in range(1, n):
            tmp=imax
            imax = max(nums[i], nums[i]*imax, nums[i]*imin)
            imin = min(nums[i], nums[i]*imin, nums[i]*tmp)
            mx = max(mx, imax)
        return mx
```


52. Max consecutive ones. https://leetcode.com/problems/max-consecutive-ones-iii/

Given an array A of 0s and 1s, we may change up to K values from 0 to 1.

Return the length of the longest (contiguous) subarray that contains only 1s. 

``` 
class Solution:
    def longestOnes(self, A: List[int], K: int) -> int:
        if not A:
            return 0
        n= len(A)
        mx = 0
        i =j =0
        while j<n and i<n:
            if A[j]==1:
                j+=1
            elif A[j]==0:
                if K>0:
                    j+=1
                    K-=1
                else:
                    mx = max(mx, j-i)
                    while K==0:
                        if A[i]==0:
                            K+=1
                        i+=1
        mx = max(mx, j-i)
        return mx
```