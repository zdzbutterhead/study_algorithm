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
'''


'''
# 使用多种基本数据类型来初始化数组
numbers1: list[int] = [0] * 5
numbers2 = list(range(5))
print(numbers1)
print(numbers2)
'''



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