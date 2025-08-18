'''
def for_loop(n):
    result = 0
    for i in range(1,n+1):
        result += i
    return result

if __name__ == '__main__':
    n = int(input("Enter a number: "))
    print(f'求和结果为：{for_loop(n)}')




def while_loop(n:int)->int:
    result = 0
    i = 1
    while i <= n:
        result += i
        i+=1
    return result

if __name__ == '__main__':
    n = int(input("Enter a number: "))
    print(f'求和结果为：{while_loop(n)}')




def while_loop_ii(n:int)->int:
    result = 0
    i = 1
    while i<= n:
        result += i
        i+=1
        i*=2
    return result




def nested_for_loop(n:int)->int:
    result = ''
    for i in range(1,n+1):
        for j in range(1,n+1):
            result += f'({i},{j}),'
    return result

if __name__ == '__main__':
    n = int(input("Enter a number: "))
    print(f'求和结果为：{nested_for_loop(n)}')




def recursion(n:int)->int:
    if n == 1:
        return 1
    else:
        return n + recursion(n-1)

if __name__ == '__main__':
    n = int(input("Enter a number: "))
    print(f'求和结果为：{recursion(n)}')




def tail_recursion(n:int, result:int)->int:
    if n == 1:
        return result
    else:
        return tail_recursion(n-1, n+result)

if __name__ == '__main__':
    n = int(input("Enter a number: "))
    print(f'求和结果为：{tail_recursion(n, 0)}')




def fib(n: int) -> int:
    """斐波那契数列：递归"""
    # 终止条件 f(1) = 0, f(2) = 1
    if n == 1 or n == 2:
        return n - 1
    # 递归调用 f(n) = f(n-1) + f(n-2)
    res = fib(n - 1) + fib(n - 2)
    # 返回结果 f(n)
    return res




def for_loop_recursion(n:int)->int:
    stack = []
    result = 0
    for i in range(n,0,-1):
        stack.append(i)
    while stack:
        result = stack.pop()




def bubble_sort(nums:list[int])->list:
    for i in range(len(nums)-1):
        for j in range(len(nums)-1-i):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
    return nums




# 使用多种基本数据类型来初始化数组
numbers1: list[int] = [0] * 5
numbers2 = list(range(5))
print(numbers1)
print(numbers2)




import sys

eng1 = "a"
eng2 = "ab"
eng3 = "abc"

chn1 = "哈"
chn2 = "哈啰"
chn2_eng1 = "哈啰a"

bmp1 = "𨊻"
bmp2 = "𨊻𨋾"
bmp2_eng1 = "𨊻𨋾a"

print("\n英文：") # 英文长度为 1
print(eng1 + ": ", sys.getsizeof(eng1))
print(eng2 + ": ", sys.getsizeof(eng2))
print(eng3 + ": ", sys.getsizeof(eng3))

print("\n中文:")
print(chn1 + ": ", sys.getsizeof(chn1))
print(chn2 + ": ", sys.getsizeof(chn2))
print(chn2_eng1 + ": ", sys.getsizeof(chn2_eng1))

print("\n补充平面:")
print(bmp1 + ": ", sys.getsizeof(bmp1))
print(bmp2 + ": ", sys.getsizeof(bmp2))
print(bmp2_eng1 + ": ", sys.getsizeof(bmp2_eng1))




import random

def random_access(nums:list[int])->int:
    return nums[random.randint(0,len(nums)-1)]

if __name__ == "__main__":
    nums = [1,2,3,4,5]
    print(random_access(nums))
    print('随机访问结果为：', random_access(nums))
    print(f'随机访问结果为：{random_access(nums)}')




def insert(nums:list[int],index:int,values:int):
    if index<0 or index>len(nums):
        raise IndexError("index out of range")
    for i in range(len(nums)-1,index,-1):
        nums[i]=nums[i-1]
    nums[index] = values

if __name__ == "__main__":
    nums = [1,2,3,4,5]
    #insert(nums,2,6)
    nums.insert(2,6)
    print(nums)




def delete(nums:list[int],index:int):
    if index<0 or index>len(nums)-1:
        raise IndexError("index out of range")
    for i in range(index,len(nums)-1):
        nums[i] = nums[i+1]

if __name__ == "__main__":
    nums = [1,2,3,4,5]
    #delete(nums,2)
    #nums.pop(2)
    #nums.remove(2)
    #del  nums[2]
    print(nums)




def traverse(nums:list[int]):
    count1 = 0
    count2 = 0
    for i in range(len(nums)):
        count1 += nums[i]
    for num in nums:
        count2 += num
    return count1,count2

if __name__ == '__main__':
    nums = [1,2,3,4,5]
    print(traverse(nums))




def find(nums:list[int],values:int):
    for i in range(len(nums)):
        if nums[i] == values:
            return i
    return -1




def extend(nums:list[int],enlarge:int):
    res = [0] * (len(nums) + enlarge)
    for i in range(len(nums)):
        res[i] = nums[i]
    return res




class ListNode:
    def __init__(self,val:int):
        self.val = val
        self.next:ListNode|None = None

def insert(n0:ListNode,P:ListNode):
    P.next = n0.next
    n0.next = P



if __name__ == '__main__':
    n0 = ListNode(1)
    n1 = ListNode(2)
    n2 = ListNode(3)
    n3 = ListNode(4)
    n4 = ListNode(5)
    n0.next = n1
    n1.next = n2
    n2.next = n3
    n3.next = n4

    p = ListNode(0)
    insert(n0,p)

    print(n0.next.val)




class MyList:
    def __init__(self):
        self._capacity: int = 10
        self._arr: list[int] = [0]*self._capacity
        self._size: int = 0
        self._extend_ratio: int = 2

    def size(self)->int:
        return self._size

    def capacity(self)->int:
        return self._capacity

    def get(self,index:int):
        if index < 0 or index >= self._size:
            raise IndexError("index out of range")
        return self._arr[index]

    def add(self,value:int):
        if self._size == self._capacity:
            self.extend_capacity()
        self._arr[self._size] = value
        self._size += 1

    def insert(self,index:int,value:int):
        if index < 0 or index >= self._size:
            raise IndexError("index out of range")
        if self._size == self._capacity:
            self.extend_capacity()
        for i in range(self._size-1,index-1,-1):
            self._arr[i+1] = self._arr[i]
        self._arr[index] = value
        self._size += 1

    def remove(self,index:int):
        if index < 0 or index >= self._size:
            raise IndexError("index out of range")
        for i in range(index,self._size-1):
            self._arr[i] = self._arr[i+1]
        self._size -= 1

    def extend_capacity(self):
        self._arr = self._arr + [0] * self.capacity() * (self._extend_ratio - 1)
        # 更新列表容量
        self._capacity = len(self._arr)

    def to_array(self) -> list[int]:
        return self._arr[:self._size]




res = [0] * 4

print(res)




class LinkedListNode:

    def __init__(self):
        self.peek :ListNode|None = None
        self.size :int = 0

    def size(self) -> int:
        return self.size

    def is_empty(self) -> bool:
        return self.size == 0

    def push(self, value: int):
        node = ListNode(value)
        node.next = self.peek
        self.peek = node
        self.size += 1

    def pop(self):
        num = self.peek()
        self.peek =self.peek.next
        self.size -= 1
        return num

    def peek(self) -> int:
        if self.is_empty():
            raise IndexError("peek from empty list")
        return self.peek.value

    def to_list(self) -> list[int]:
        res = []
        node = self.peek
        while node:
            res.append(node.value)
            node = node.next
        return res




class ArrayStack:

    def __init__(self):
        self._stack:list[int]=[]

    def size(self) -> int:
        return len(self.stack)

    def is_empty(self) ->bool:
        return self.size() == 0

    def push(self,item):
        self._stack.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._stack.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._stack[-1]

    def to_list(self):
        return self._stack




from collections import deque
que:deque[int] = deque()

que.append(1)
que.append(3)
que.append(2)
que.append(5)
que.append(4)

front:int = que[0]

pop :int =que.left.pop()

size:int = len(que)

is_empty:bool = len(que) == 0




class LinkedListQueue:

    def __init__(self):
        self._front:LinkedListNode|None = None
        self._rear:LinkedListNode|None = None
        self._size:int = 0

    def size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def push(self, value: int):
        node = ListNode(value)
        if self._front is None:
            self._front = node
            self._rear = node
        else:
            self._rear.next = node
            self._rear = node
        self._size += 1

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty queue")
        return self._front.value

    def to_list(self):
        queue = []
        temp = self._front
        while temp:
            queue.append(temp.value)
            temp = temp.next
        return queue




class ArrayQueue:

    def __init__(self,size:int):
        self._nums: list[int] = [0] * size
        self._front: int = 0
        self._size: int = 0

    def capacity(self)->int:
        return len(self._nums)

    def size(self):
        return self.size()

    def is_empty(self):
        return self.size() == 0

    def push(self,value:int):
        if self.size() == self.capacity():
            raise IndexError("queue is full")
        rear :int = (self._front + self._size)%self.capacity()
        self._nums[rear] = num
        self._size += 1

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty queue")
        num = self._nums[self._front]
        self._front = (self._front + 1) % self.capacity()
        self._size -= 1
        return num

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self._nums[self._front]

    def to_list(self):
        res = [0]*self.size()
        j =self._front
        for i in range(self.size()):
            res[i] = self.nums[(j%self.capacity())]
            j=j+1
        return res




from collections import deque

# 初始化双向队列
deq: deque[int] = deque()

# 元素入队
deq.append(2)      # 添加至队尾
deq.append(5)
deq.append(4)
deq.appendleft(3)  # 添加至队首
deq.appendleft(1)

# 访问元素
front: int = deq[0]  # 队首元素
rear: int = deq[-1]  # 队尾元素

# 元素出队
pop_front: int = deq.popleft()  # 队首元素出队
pop_rear: int = deq.pop()       # 队尾元素出队

# 获取双向队列的长度
size: int = len(deq)

# 判断双向队列是否为空
is_empty: bool = len(deq) == 0




class ListNode:

    def __init__(self, val=0):
        self.value =val
        self.next: ListNode | None = None
        self.prev: ListNode | None = None

class LinkedListQueue:

    def __init__(self):
        self._front: ListNode | None = None
        self._rear: ListNode | None = None
        self._size: int = 0

    def size(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def push(self, value: int,is_front:bool):
        node = ListNode(value)
        if self.is_empty():
            self._front = node
            self._rear = node
        elif is_front:
            node.next = self._front
            self._front.prev = node
            self._front = node
        else:
            self._rear.next = node
            node.prev = self._rear
            self._rear = node
        self._size += 1

    def push_first(self, value: int):
        self.push(value,True)

    def push_last(self,value: int):
        self.push(value,False)

    def pop(self,is_front:bool):
        if self.is_empty():
            raise IndexError('pop from empty queue')
        if is_front:
            value = self._front.value
            fnext: ListNode | None = self._front.next
            if fnext is not None:
                fnext.prev = None
                self._front.next = None
            self._front = fnext
        else:
            value = self._rear.value
            rprev: ListNode | None = self._rear.prev
            if rprev is not None:
                rprev.next = None
                self._rear.prev = None
            self._rear = rprev
        self._size -= 1
        return value

    def pop_first(self):
        return self.pop(True)

    def pop_last(self):
        return self.pop(False)

    def peek_first(self):
        if self.is_empty():
            raise IndexError('peek from empty queue')
        return self._front.value

    def peek_last(self):
        if self.is_empty():
            raise IndexError('peek from empty queue')
        return self._rear.value

    def to_list(self):
        res = []*self.size()
        node = self._front
        while node is not None:
            res.append(node.value)
            node = node.next
        return res




class ArrayDeque:

    def __init__(self,capacity:int):
        self._nums: list[int] = [0]*capacity
        self._front: int = 0
        self._size: int = 0

    def capacity(self):
        return len(self._nums)

    def size(self):
        return self._size

    def is_empty(self):
        return self.size() == 0

    def index(self,index:int):
        return (self._front + index)%self.capacity()

    def push_front(self,value:int):
        if self.size() == self.capacity():
            raise IndexError('queue is full')
        self._front = self.index(self._front - 1)
        self._nums[self._front] = value
        self._size += 1

    def push_last(self,value:int):
        if self.size() == self.capacity():
            raise IndexError('queue is full')
        self._nums[self.index(self._front + self._size)] = value
        self._size += 1

    def pop_first(self):
        num = self.peek_first()
        self._front = self.index(self._front + 1)
        self._size -= 1
        return num

    def pop_last(self):
        num = self.peek_last()
        self._size -= 1
        return num

    def peek_first(self):
        if self.is_empty():
            raise IndexError('peek from empty queue')
        return self._nums[self._front]

    def peek_last(self):
        if self.is_empty():
            raise IndexError('peek from empty queue')
        return self._nums[self.index(self._front + self._size - 1)]

    def to_list(self):
        return self._nums[self.index(self._front): self.index(self._front + self._size)]




hmap: dict = {}

hmap[12836] = "小哈"
hmap[15937] = "小啰"
hmap[16750] = "小算"
hmap[13276] = "小法"
hmap[10583] = "小鸭"

# 查询操作
# 向哈希表中输入键 key ，得到值 value
name: str = hmap[15937]

# 删除操作
# 在哈希表中删除键值对 (key, value)
hmap.pop(10583)

print(name)




class Pair:

    def __init__(self,key:int,value:int):
        self.key = key
        self.value = value

class ArrayHashMap:

    def __init__(self):
        self.buckets: list[Pair | None] = [None]*100

    def hash_func(self, key: int) -> int:
        index: int = key % 100
        return index

    def get(self,key:int) -> str:
        index: int = self.hash_func(key)
        pair: Pair | None = self.buckets[index]
        if pair is None:
            return None
        return pair.value

    def put(self, key: int, value: int):
        pair = Pair(key,value)
        index: int = self.hash_func(key)
        self.buckets[index] = pair

    def remove(self, key: int):
        index: int = self.hash_func(key)
        self,buckets[index] = None

    def entry_set(self) -> list[Pair]:
        return [pair for pair in self.buckets if pair is not None]

    def key_set(self):
        return [pair.key for pair in self.entry_set()]

    def value_set(self):
        return [pair.value for pair in self.entry_set()]

    def print(self):
        print(self.key_set())




class HashMapChaining:

    def __init__(self):
        self.size = 0
        self.capacity = 4
        self.load_threshold = 2.0/3.0
        self.extend_ration = 2
        self.buckets = [[] for _ in range(self.capacity)]

    def hash_func(self, key: int) -> int:
        return key%self.capacity

    def load_factor(self) -> float:
        return self.size/self.capacity

    def get(self, key: int):
        index: int = self.hash_func(key)
        bucket = self.buckets[index]
        for pair in bucket:
            if pair.key == key:
                return pair.value
        return None

    def put(self, key, value):
        if self.load_factor() > self.load_threshold:
            self.extend()
        index = self,hash_func(key)
        bucket = self.buckets[index]
        for pair in bucket:
            if pair.key == key:
                pair.value = value
                return
        pair = Pair(key,value)
        bucket.append(pair)
        self.size += 1

    def remove(self, key):
        index = self.hash_func(key)
        bucket = self.buckets[index]
        for pair in bucket:
            if pair.key == key:
                bucket.remove(pair)
                self.size -= 1
                break

    def extend(self):
        buckets = self.buckets
        self.capacity *= self.extend_ration
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for bucket in buckets:
            for pair in bucket:
                self.put(pair.key, pair.value)

    def print(self):
        for bucket in buckets:
            res = []
            for pair in bucket:
                res.append(str(pair.key) + ": " + str(pair.value))
            print(res)




class HashMapOpenAddressing:

    def __init__(self):
        self.size = 0
        self.capacity = 4
        self.load_threshold = 2.0/3.0
        self.extend_ration = 2
        self.buckets = [None] * self.capacity
        self.TOMBSTONE = Pair(-1, '-1')

    def hash_func(self, key):
        return key % self.capacity

    def load_factor(self):
        return self.size / self.capacity

    def find_bucket(self, key):
        index = self.hash_func(key)
        first_tombstone = -1
        while self.buckets[index] is not None:
            if self.buckets[index].key == key:
                if first_tombstone != -1:
                    self.buckets[first_tombstone] = self.buckets[index]
                    self.buckets[index] = self.TOMBSTONE
                return index
            if first_tombstone == -1 and self.buckets[index] is self.TOMBSTONE:
                first_tombstone = index
            index = (index + 1) % self.capacity
        return index if first_tombstone == -1 else first_tombstone

    def get(self,key):
        index = self.find_bucket(key)
        if self.buckets[index] not in [None, self.TOMBSTONE]:
            return self.buckets[index].value
        return None

    def put(self, key, value):
        if self.load_factor() > self.load_threshold:
            self.extend()
        index = self.find_bucket(key)
        if self.buckets[index] not in [None,self.TOMBSTONE]:
            self.buckets[index].value = value
            return
        self.buckets[index] = Pair(key, value)
        self.size += 1

    def remove(self, key):
        index = self.find_bucket(key)
        if self.buckets[index] not in [None, self.TOMBSTONE]:
            self.buckets[index] = self.TOMBSTONE
            self.size -= 1

    def extend(self):
        buckets_temp = self.buckets
        self.capacity *= self.extend_ration
        self.buckets = [None] * self.capacity
        self.size = 0
        for pair in buckets_temp:
            if pair is not None and pair is not self.TOMBSTONE:
                self.put(pair.key, pair.value)

    def print(self):
        """打印哈希表"""
        for pair in self.buckets:
            if pair is None:
                print("None")
            elif pair is self.TOMBSTONE:
                print("TOMBSTONE")
            else:
                print(pair.key, "->", pair.val)




def add_hash(key:str) -> int:
    hash = 0
    modulus = 1000000007
    for c in key:
        hash += ord(c)
    return hash % modulus

def mul_hash(key:str) -> int:
    hash = 0
    modulus = 1000000007
    for c in key:
        hash = 31*hash + ord(c)
    return hash % modulus

def xor_hash(key:str) -> int:
    hash = 0
    modulus = 1000000007
    for c in key:
        hash ^= ord(c)
    return hash % modulus

def rot_hash(key:str) -> int:
    hash = 0
    modulus = 1000000007
    for c in key:
        hash = (hash << 4) ^ (hash >> 28) ^ ord(c)
    return hash % modulus




def level_order(root: TreeNode | None) -> list[int]:
    queue: deque[TreeNode] = deque()
    queue.append(root)
    res = []
    while queue:
        node = queue.popleft()
        res.append(node.val)
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)
    return res




def pre_order(root: TreeNode | None):
    if root is None:
        return
    res.append(root.val)
    pre_order(root.left)
    pre_order(root.right)

def in_order(root: TreeNode | None):
    if root is None:
        return
    in_order(root.left)
    res.append(root.val)
    in_order(root.right)

def post_order(root: TreeNode | None):
    if root is None:
        return
    post_order(root.left)
    post_order(root.right)
    res.append(root.val)



class TreeNode:
    """二叉树节点类"""

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def preorder_traversal(root: TreeNode):
    """前序遍历：根 -> 左 -> 右（非递归）"""
    if not root:
        return []

    result = []
    stack = [root]  # 栈初始化，先放入根节点

    while stack:
        node = stack.pop()  # 弹出栈顶节点
        result.append(node.val)  # 访问根节点

        # 注意：栈是后进先出，所以先放右子树，再放左子树
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result


def inorder_traversal(root: TreeNode):
    """中序遍历：左 -> 根 -> 右（非递归）"""
    if not root:
        return []

    result = []
    stack = []
    current = root

    while current or stack:
        # 先遍历左子树，将所有左节点入栈
        while current:
            stack.append(current)
            current = current.left

        # 弹出栈顶节点（最左节点）
        current = stack.pop()
        result.append(current.val)  # 访问根节点

        # 转向右子树
        current = current.right

    return result


def postorder_traversal(root: TreeNode):
    """后序遍历：左 -> 右 -> 根（非递归）"""
    if not root:
        return []

    result = []
    stack = [root]
    visited = set()  # 记录已访问的节点

    while stack:
        node = stack[-1]  # 查看栈顶节点（不弹出）

        # 如果节点的左右子树都已访问，或者是叶子节点，则访问该节点
        if (not node.left and not node.right) or node in visited:
            stack.pop()
            result.append(node.val)
        else:
            # 标记当前节点为已访问（表示其左右子树即将被处理）
            visited.add(node)
            # 先放右子树，再放左子树（栈是后进先出）
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)

    return result




class ArrayBinaryTree:

    def __init__(self, arr: list[int]):
        self._tree = list(arr)

    def size(self):
        return len(self._tree)

    def val(self,i: int) -> int | None:
        if i < 0 or i >= self.size():
            return None
        return self._tree[i]

    def left(self, i: int) -> int | None:
        return 2 * i + 1

    def right(self, i: int) -> int | None:
        return 2 * i + 2

    def parent(self, i: int) -> int | None:
        return (i - 1) // 2

    def level_order(self) -> list[int]:
        self.res = []
        for i in range(self.size()):
            if self.val(i) is not None:
                self.res.append(self.val(i))
        return self.res

    def dfs(self, i: int, order: str):
        if self.val(i) is None:
            return
        if order == "pre":
            self.res.append(self.val(i))
        self.dfs(self.left(i), order)
        if order == "in":
            self.res.append(self.val(i))
        self.dfs(self.right(i),order)
        if order == "post":
            self.res.append(self.val(i))

    def pre_order(self) -> list[int]:
        self.res = []
        self.dfs(0, order = "pre")
        return self.res

    def in_order(self) -> list[int]:
        self.res = []
        self.dfs(0, order = "in")
        return self.res

    def post_order(self) -> list[int]:
        self.res = []
        self.dfs(0, order = "post")
        return self.res




def remove(self, num: int):
    if self._root is None:
        return
    cur, pre = self._root, num
    while cur is not None:
        if cur.val == num:
            break
        pre = cur
        if cur.val < num:
            cur = cur.right
        else:
            cur = cur.left

    if cur is None:
        return

    if cur.left is None or cur.right is None:
        child = cur.left or cur.right
        if cur != self._root:
            if pre.left == cur:
                pre.left = child
            else:
                pre.right = child
        else:
            self._root = child
    else:
        tmp = cur.right
        while tmp.left is not None:
            tmp = tmp.left
        self.remove(tmp.val)
        cur.val = tmp.val




class TreeNode:

    def __init__(self,val: int):
        self.val: int = val
        self.height: int = 0
        self.left: TreeNode | None = None
        self.right: TreeNode | None = None

    def height(self, node: TreeNode | None):
        if node is None:
            return -1
        return node.hegiht

    def update_height(self, node: TreeNode | None):
        node.height = max([self.height(node.left), self.height(node.right)])+1

    def balance_factor(self, node: TreeNode | None) -> int:
        if node is None:
            return 0
        return self.height(node.left)-self.height(node.right)

    def right_rotate(self, node: TreeNode | None) -> TreeNode | None:
        child = node.left
        grand_child = child.right
        child.right = node
        node.left = grand_child
        self.update_height(child)
        self.updata_height(node)
        return child

    def left_rotate(self, node: TreeNode | None) -> TreeNode | None:
        child = node.right
        grand_child = child.left
        child.left = node
        node.right = grand_child
        self.update_height(child)
        self.update_height(node)
        return child

    def rotate(self, node: TreeNode | None) -> TreeNode | None:
        balance_factor = self.balcance_factor(node)
        if balance_factor > 1:
            if self.balance_factor(node.left) >= 0:
                return self.right_rotate(node)
            else:
                node.left = self.left_rotate(node.left)
                return self.right_rotate(node)
        elif balance_factor < -1:
            if self.balance_factor(node.right) <= 0:
                return self.left_rotate(node)
            else:
                node.right = self.right_rotate(node.right)
                return self.left_rotate(node)
        return node

    def insert(self, val):
        self._root = self.insert_helper(self._root,val)

    def insert_helper(self, node: TreeNode | None, val: int) -> TreeNode:
        if node is None:
            return TreeNode(val)
        if val < node.val:
            node.left = self.insert_helper(node.left, val)
        elif val > node.val:
            node.right = self.insert_helper(node.right, val)
        else:
            return node
        self.update_height(node)
        return self.rotate(node)

    def remove(self, node: TreeNode | None, val: int) -> TreeNode | None:
        if node is None:
            return None
        if val < node.val:
            node.left = self.remove(node.left, val)
        elif val > node.val:
            node.right = self.remove(node.right, val)
        else:
            if node.left is None or node.right is None:
                child = node.left or node.right
                if child is None:
                    return None
                else:
                    node = child
            else:
                temp = node.right
                while temp is not None:
                    temp = temp.left
                node.right = self.remove(node.right, temp.val)
                node.val = temp.val
        self.update_height(node)
        return self.rotate(node)
        '''



import heapq

min_heap, flag = [], 1

max_heap, flag = [], -1

heapq.heappush(max_heap, flag * 1)
heapq.heappush(max_heap, flag * 2)
heapq.heappush(max_heap, flag * 2)
heapq.heappush(max_heap, flag * 5)
heapq.heappush(max_heap, flag * 4)

peek: int = flag * max_heap[0] # 5

val = flag * heapq.heappop(max_heap) # 5
val = flag * heapq.heappop(max_heap) # 4
val = flag * heapq.heappop(max_heap) # 3
val = flag * heapq.heappop(max_heap) # 2
val = flag * heapq.heappop(max_heap) # 1

size: int = len(max_heap)

# 判断堆是否为空
is_empty: bool = not max_heap

# 输入列表并建堆
min_heap: list[int] = [1, 3, 2, 5, 4]
heapq.heapify(min_heap)




def leaf(self, i: int):
    return 2 * 1 + 1

def right(self, i: int):
    return 2 * i + 2

def parent(self, i: int):
    return (i - 1) // 2

def peek(self) -> int:
    return self.max_heap[0]




def push(self, value: int):
    self.max_heap.append(value)
    self.sift_up(self.size()-1)

def sift_up(self, i: int):
    while True:
        parent_index = self.parent(i)
        if parent_index < 0 or self.max_heap[parent_index] >= self.max_heap[i]:
            break
        self.swap(i,parent_index)
        i = parent_index




def pop(self) -> int:
    if self.is_empty():
        raise IndexError("pop from an empty heap")
    self.swap(0, self.size()-1)
    value = self.max_heap.pop()
    self.sift_down(0)
    return value

def sift_down(self,i: int):
    while True:
        left,right,max = self.left(i),self.right(i),i
        if left < self.size() and self.max_heap[left] < self.max_heap[max]:
            max = left
        if right < self.size() and self.max_heap[right] <self.max_heap[max]:
            max = right
        if max == i:
            break
        self.swap(i,max)
        i = max

















