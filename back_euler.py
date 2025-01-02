import numpy as np
'''
f  : 函数 f(x, y)，表示微分方程右侧
y0 : 初值 y(0)
y1 : 初值 y(0.1)
x0 : 初始 x 值
h  : 步长
n  : 迭代步数
'''
def backward_euler(f, y0, x0, h, n):

    x_vals = [x0]
    y_vals = [y0]
    
    for i in range(n):
        x_next = x_vals[-1] + h
        y_prev = y_vals[-1]
        
        # 求解 y_next
        y_next = y_prev  # 初始猜测值
        for _ in range(10):  # 迭代 10 次
            y_next = y_prev + h * f(x_next, y_next)
        
        x_vals.append(x_next)
        y_vals.append(y_next)
    
    return x_vals, y_vals

# 定义微分方程 
def f(x, y):
    return x**2 + x - y

# 初值和参数
y0 = 0     
x0 = 0      
h = 0.01   
n = 10       

# 调用
x_vals, y_vals = backward_euler(f, y0, x0, h, n)

# 输出结果
for i in range(len(x_vals)):
    print(f"x = {x_vals[i]:.2f}, y = {y_vals[i]:.5f}")

print(f"y(0.1) ≈ {y_vals[-1]:.5f}")