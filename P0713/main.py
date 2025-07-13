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
'''



