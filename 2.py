import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return 0.5 * (-y + x**2 + 4*x - 1)

def exact_solution(x):
    return np.exp(-x / 2) + x**2 - 1

# 改进欧拉法 
def improved_euler(h, x0, y0, x_end):  
    x_vals = [x0]  
    y_vals = [y0]  
    while x_vals[-1] < x_end:  
        x_n = x_vals[-1]  
        y_n = y_vals[-1]  
        x_next = x_n + h  
        
        # 预测  
        y_pred = y_n + h * f(x_n, y_n)  
        
        # 校正  
        y_next = y_n + (h / 2) * (f(x_n, y_n) + f(x_next, y_pred))  
        
        x_vals.append(x_next)  
        y_vals.append(y_next)  
    return np.array(x_vals), np.array(y_vals)

# 三阶 Runge-Kutta 方法
def runge_kutta_3(h, x0, y0, x_end):
    x_vals = [x0]
    y_vals = [y0]
    while x_vals[-1] < x_end:
        x_n = x_vals[-1]
        y_n = y_vals[-1]
        k1 = f(x_n, y_n)
        k2 = f(x_n + h / 2, y_n + h / 2 * k1)
        k3 = f(x_n + h, y_n - h * k1 + 2 * h * k2)
        x_next = x_n + h
        y_next = y_n + h / 6 * (k1 + 4 * k2 + k3)
        x_vals.append(x_next)
        y_vals.append(y_next)
    return np.array(x_vals), np.array(y_vals)

# 四阶 Runge-Kutta 方法
def runge_kutta_4(h, x0, y0, x_end):
    x_vals = [x0]
    y_vals = [y0]
    while x_vals[-1] < x_end:
        x_n = x_vals[-1]
        y_n = y_vals[-1]
        k1 = f(x_n, y_n)
        k2 = f(x_n + h / 2, y_n + h / 2 * k1)
        k3 = f(x_n + h / 2, y_n + h / 2 * k2)
        k4 = f(x_n + h, y_n + h * k3)
        x_next = x_n + h
        y_next = y_n + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_vals.append(x_next)
        y_vals.append(y_next)
    return np.array(x_vals), np.array(y_vals)

# 误差计算
def compute_error(y_num, y_exact):
    return np.sqrt(np.mean((y_num - y_exact)**2))

h_values = [0.1, 0.01, 0.001, 0.0001]
errors_euler = []
errors_rk3 = []
errors_rk4 = []

for h in h_values:
    x0, y0, x_end = 0, 0, 1
    # 改进欧拉法
    x_euler, y_euler = improved_euler(h, x0, y0, x_end)
    y_exact = exact_solution(x_euler)
    errors_euler.append(compute_error(y_euler, y_exact))
    
    # 三阶 Runge-Kutta
    x_rk3, y_rk3 = runge_kutta_3(h, x0, y0, x_end)
    y_exact = exact_solution(x_rk3)
    errors_rk3.append(compute_error(y_rk3, y_exact))
    
    # 四阶 Runge-Kutta
    x_rk4, y_rk4 = runge_kutta_4(h, x0, y0, x_end)
    y_exact = exact_solution(x_rk4)
    errors_rk4.append(compute_error(y_rk4, y_exact))

# 对比图
x_vals = np.linspace(0, 1, 100)
y_exact = exact_solution(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_exact, label="Exact Solution", linestyle='--')
plt.plot(x_euler, y_euler, label="Improved Euler Method", marker='o')
plt.plot(x_rk3, y_rk3, label="Runge-Kutta 3rd Order", marker='s')
plt.plot(x_rk4, y_rk4, label="Runge-Kutta 4th Order", marker='^')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Numerical Methods and Exact Solution")
plt.legend()
plt.grid()
plt.show()

# 误差对比图
log_h = np.log10(h_values)
log_error_euler = np.log10(errors_euler)
log_error_rk3 = np.log10(errors_rk3)
log_error_rk4 = np.log10(errors_rk4)

plt.figure(figsize=(10, 6))
plt.plot(log_h, log_error_euler, label="Improved Euler Method", marker='o')
plt.plot(log_h, log_error_rk3, label="Runge-Kutta 3rd Order", marker='s')
plt.plot(log_h, log_error_rk4, label="Runge-Kutta 4th Order", marker='^')
plt.xlabel("log(h)")
plt.ylabel("log(Error)")
plt.title("Error Analysis of Numerical Methods")
plt.legend()
plt.grid()
plt.show()