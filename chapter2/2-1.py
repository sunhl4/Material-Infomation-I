import jax.numpy as jnp
from jax import grad, jit, vmap
from matplotlib import pyplot as plt
# 从表格中提取的原始数据

temperatures_celsius = jnp.array([25, 50, 70, 80])  # 摄氏度
D_values_group_I = jnp.array([4.11, 4.92, 6.10, 6.66])  # Group I
D_values_group_II = jnp.array([4.27, 4.92, 6.14, 6.62])  # Group II
# 将摄氏度转换为开尔文
temperatures_kelvin = temperatures_celsius + 273.15
# 计算 1/T (10^-3 K^-1)
x_data_group_I = 1e3 / temperatures_kelvin
x_data_group_II = 1e3 / temperatures_kelvin
# 将 D 值转换为自然对数
y_data_group_I = jnp.log(D_values_group_I) + jnp.log(1E-5)
y_data_group_II = jnp.log(D_values_group_II) + jnp.log(1E-5)
# 组合两组数据
x_data_combined = jnp.concatenate([x_data_group_I, x_data_group_II])
y_data_combined = jnp.concatenate([y_data_group_I, y_data_group_II])

# 对每组数据单独拟合
params_group_I = jnp.array([1.0, 1.0])
params_group_II = jnp.array([1.0, 1.0])
# 定义线性模型
@jit
def model(params, x):
    return params[0] * x + params[1]

# 损失函数：最小二乘法
@jit
def loss(params, x, y):
    return jnp.mean((model(params, x) - y) ** 2)

# 梯度下降函数
@jit
def update(params, x, y, lr=0.01):
    gradients = grad(loss)(params, x, y)
    return params - lr * gradients

for _ in range(100000):
    params_group_I = update(params_group_I, x_data_group_I, y_data_group_I)
    params_group_II = update(params_group_II, x_data_group_II, y_data_group_II)

# 使用最终参数计算拟合线
x_range = jnp.linspace(x_data_combined.min(), x_data_combined.max(), 100)
fit_line_group_I = model(params_group_I, x_range)
fit_line_group_II = model(params_group_II, x_range)

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(x_data_group_I, y_data_group_I, marker='o', label='Group I samples')
plt.scatter(x_data_group_II, y_data_group_II, marker='^', label='Group II samples')
plt.plot(x_range, fit_line_group_I, linestyle='--', label='Group I fit')
plt.plot(x_range, fit_line_group_II, linestyle=':', label='Group II fit')

plt.xlabel('1/T ($10^{-3}$ K$^{-1}$)')
plt.ylabel('ln[D($cm^{2}$ s$^{-1}$)]')
plt.title('The LSLR of lnD versus 1/T')
plt.legend()
plt.show()
