import numpy as np
'''
f  : 函数 f(x, y)，表示微分方程右侧
y0 : 初值 y(0)
y1 : 初值 y(0.1)
x0 : 初始 x 值
h  : 步长
n  : 迭代步数
'''
def two_step_euler(f, y0, x0, h, n):

    x_vals = [x0]
    y_vals = [y0]
    
    for i in range(n):
        x_n = x_vals[-1]
        y_n = y_vals[-1]
        
        # 第一步：显式欧拉法（预测值）
        y_predict = y_n + h * f(x_n, y_n)
        
        # 第二步：隐式欧拉法（校正值）
        x_next = x_n + h
        y_next = y_n + (h / 2) * (f(x_n, y_n) + f(x_next, y_predict))
        
        x_vals.append(x_next)
        y_vals.append(y_next)
    
    return x_vals, y_vals

# 定义微分方程 y'(x) = x^2 + x - y
def f(x, y):
    return x**2 + x - y

# 定义精确解 y(x) = -e^(-x) + x^2 - x + 1
def exact_solution(x):
    return -np.exp(-x) + x**2 - x + 1

# 初值和参数
y0 = 0.00566  
x0 = 0.1    
h = 0.1      
n = 9        

# 调用
x_vals, y_vals = two_step_euler(f, y0, x0, h, n)

# 计算
exact_vals = [exact_solution(x) for x in x_vals]

print("x\t\tNumerical y\tExact y\t\tError")
for i in range(len(x_vals)):
    print(f"{x_vals[i]:.1f}\t{y_vals[i]:.5f}\t{exact_vals[i]:.5f}\t{abs(y_vals[i] - exact_vals[i]):.5e}")

print(f"\ny(1) ≈ {y_vals[-1]:.5f}, 精确解 y(1) = {exact_vals[-1]:.5f}, 误差 = {abs(y_vals[-1] - exact_vals[-1]):.5e}")