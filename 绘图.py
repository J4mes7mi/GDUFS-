import numpy as np  
import matplotlib.pyplot as plt  

# 定义解析解  
def exact_solution(x):  
    return -np.exp(-x) + x**2 - x + 1  

# 定义 x 值范围  
x_vals = np.linspace(0, 1, 11)  

# 数值解（假设已计算得到）  
improved_vals = [0,0.00566, 0.02207, 0.05028, 0.09105, 0.14510, 0.21307, 0.29552, 0.39300, 0.50596, 0.63485]  
predictor_vals = [0.00000, 0.00566, 0.02183, 0.04978, 0.09029, 0.14409, 0.21307, 0.29424, 0.39129, 0.50403, 0.63271]
predictor_corrector_vals = [0.00000, 0.00516, 0.02164, 0.04951, 0.08998, 0.14374, 0.21143, 0.29363, 0.39086, 0.50360, 0.63227] 
# 解析解  
exact_vals = [exact_solution(x) for x in x_vals]  

# 绘图  
plt.figure(figsize=(10, 6))  

plt.plot(x_vals, improved_vals, label="Improved Two-Step Euler Method", marker='s')  
plt.plot(x_vals, predictor_vals, label="Predictor-Corrector Method", marker='^')  
plt.plot(x_vals, predictor_corrector_vals, label="Predictor-Corrector System", marker='o')  
plt.plot(x_vals, exact_vals, label="Exact Solution", linestyle='--')  

# 图例和标签  
plt.xlabel("x")  
plt.ylabel("y")  
plt.title("Comparison of Numerical Methods and Exact Solution")  
plt.legend()  
plt.grid()  
plt.show()