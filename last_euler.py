import numpy as np

def predictor_corrector_system(f, y0, y1, x0, h, n):
    """
    f  : 函数 f(x, y)，表示微分方程右侧
    y0 : 初值 y(0)
    y1 : 初值 y(0.1)
    x0 : 初始 x 值
    h  : 步长
    n  : 迭代步数
    
    x_vals : x 的值列表
    y_vals : y 的值列表
    """
    x_vals = [x0, x0 + h]
    y_vals = [y0, y1]
    y_prime_vals = [f(x0, y0), f(x0 + h, y1)]
    c_vals = [y0, y1]
    p_vals = [y0, y1]

    for i in range(1, n):
        x_n_minus_1 = x_vals[i - 1]
        x_n = x_vals[i]
        y_n_minus_1 = y_vals[i - 1]
        y_n = y_vals[i]
        y_prime_n = y_prime_vals[i]
        c_n = c_vals[i]
        p_n = p_vals[i]
        # 第一步：预测
        p_next = y_n_minus_1 + 2 * h * y_prime_n
        
        # 第二步：改进预测
        m_next = p_next - (4 / 5) * (p_n - c_n)
        
        # 第三步：计算 m'_{n+1}
        m_prime_next = f(x_n + h, m_next)
        
        # 第四步：校正
        c_next = y_n + (h / 2) * (m_prime_next + y_prime_n)
        
        # 第五步：改进校正
        y_next = c_next + (1 / 5) * (p_next - c_next)
        
        # 第六步：计算 y'_{n+1}
        y_prime_next = f(x_n + h, y_next)
        
        # 更新值
        x_vals.append(x_n + h)
        y_vals.append(y_next)
        y_prime_vals.append(y_prime_next)
        c_vals.append(c_next)
        p_vals.append(p_next)  # 更新 p_n 为 p_next  
    return x_vals, y_vals

# 定义微分方程 y'(x) = x^2 + x - y
def f(x, y):
    return x**2 + x - y

# 定义精确解 y(x) = -e^(-x) + x^2 - x + 1
def exact_solution(x):
    return -np.exp(-x) + x**2 - x + 1

# 初值和参数
y0 = 0       
y1 = 0.00566 
x0 = 0       
h = 0.1     
n = 10      

# 调用系统
x_vals, y_vals = predictor_corrector_system(f, y0, y1, x0, h, n)

# 计算精确解
exact_vals = [exact_solution(x) for x in x_vals]


print("x\t\tNumerical y\tExact y\t\tError")
for i in range(len(x_vals)):
    print(f"{x_vals[i]:.1f}\t{y_vals[i]:.5f}\t{exact_vals[i]:.5f}\t{abs(y_vals[i] - exact_vals[i]):.5e}")

print(f"\ny(1) ≈ {y_vals[-1]:.5f}, 精确解 y(1) = {exact_vals[-1]:.5f}, 误差 = {abs(y_vals[-1] - exact_vals[-1]):.5e}")