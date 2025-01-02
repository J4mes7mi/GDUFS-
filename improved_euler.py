import numpy as np
'''
f  : 函数 f(x, y)，表示微分方程右侧
y0 : 初值 y(0)
y1 : 初值 y(0.1)
x0 : 初始 x 值
h  : 步长
n  : 迭代步数
'''


def improved_two_step_euler(f, y0, y1, x0, h, n):
    x_vals = [x0, x0 + h]
    y_vals = [y0, y1]
    
    for i in range(1, n):
        x_n_minus_1 = x_vals[i - 1]
        x_n = x_vals[i]
        y_n_minus_1 = y_vals[i - 1]
        y_n = y_vals[i]
        
        # 第一步：预测公式
        x_next = x_n + h
        y_predict = y_n_minus_1 + 2 * h * f(x_n, y_n)
        
        # 第二步：校正公式
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
y0 = 0       # y(0) = 0
y1 = 0.00566 # y(0.1) = 0.00566
x0 = 0       # x 起始值
h = 0.1      # 步长
n = 10       # 计算到 x = 1，共 10 步

# 调用改进两步欧拉法
x_vals, y_vals = improved_two_step_euler(f, y0, y1, x0, h, n)

# 计算精确解
exact_vals = [exact_solution(x) for x in x_vals]

# 输出结果
print("x\t\tNumerical y\tExact y\t\tError")
for i in range(len(x_vals)):
    print(f"{x_vals[i]:.1f}\t{y_vals[i]:.5f}\t{exact_vals[i]:.5f}\t{abs(y_vals[i] - exact_vals[i]):.5e}")

# 打印 y(1) 的结果
print(f"\ny(1) ≈ {y_vals[-1]:.5f}, 精确解 y(1) = {exact_vals[-1]:.5f}, 误差 = {abs(y_vals[-1] - exact_vals[-1]):.5e}")