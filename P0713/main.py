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
    '''





