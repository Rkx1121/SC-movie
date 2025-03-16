import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 设置字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows 中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 定义矩阵数据
matrix = np.array([
    [1, 0.24, 0.87, 0.64, 0.87, 0.32],
    [0.24, 1, 0.46, 0.21, 0.46, 0.14],
    [0.87, 0.46, 1, -0.16, 1, 0.8],
    [0.64, 0.21, -0.16, 1, 0.88, 0.66],
    [0.87, 0.46, 1, 0.88, 1, 0.73],
    [0.32, 0.14, 0.8, 0.66, 0.73, 1]
])

# 定义标签
labels = ["累计票房（亿）", "科技厅票房占比", "排片占比", "场均人次", "排座占比", "黄金占比"]

# 创建 fig 文件夹
output_dir = "fig"
os.makedirs(output_dir, exist_ok=True)

# 画热力图（使用红色系）
plt.figure(figsize=(10, 6), dpi=300)  # 设置高清分辨率
sns.heatmap(matrix, annot=True, cmap="Reds", xticklabels=labels, yticklabels=labels, fmt=".2f", linewidths=0.5)

# 添加标题
plt.title("票房与特征的相关系数", fontsize=14)

# 保存图片
save_path = os.path.join(output_dir, "票房与特征的相关系数.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

# 显示图像
# plt.show()


