import numpy as np  
import matplotlib.pyplot as plt  
from scipy.integrate import quad  

# 定义积分 
def g(x):  
    return np.sqrt(1 + 4 * x**2)  

# 复合Simpson 
def composite_simpson(a, b, m):  
    h = (b - a) / m  
    x = np.linspace(a, b, m + 1)  
    y = g(x)  
    S = h / 3 * (y[0] + 4 * np.sum(y[1:m:2]) + 2 * np.sum(y[2:m-1:2]) + y[m])  
    return S  

# 精确解  
a, b = 0, 1  
exact_value, _ = quad(g, a, b)  

# 计算不同步长下的误差  
h_values = [1, 1/2, 1/4, 1/8, 1/16, 1/28]  
errors = []  

for h in h_values:  
    m = int((b - a) / h)  
    approx_value = composite_simpson(a, b, m)  
    error = abs(approx_value - exact_value)  
    errors.append(error)  

print("步长 h\t\t近似值\t\t误差")  
for h, error in zip(h_values, errors):  
    print(f"{h:.5f}\t{composite_simpson(a, b, int((b - a) / h)):.8f}\t{error:.8e}")  

# 绘图  
log_h = np.log10(h_values)  
log_error = np.log10(errors)  

plt.figure(figsize=(8, 6))  
plt.plot(log_h, log_error, marker='o', label="Error")  
plt.xlabel("log(h)")  
plt.ylabel("log(Error)")  
plt.title("log(h) vs log(Error)")  
plt.grid()  
plt.legend()  
plt.show()  

# 斜率图  
slope = np.polyfit(log_h, log_error, 1)[0]  
print(f"误差的 log-log 图斜率为：{slope:.2f}")