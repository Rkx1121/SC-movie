import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于 Windows
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 确保 'fig' 文件夹存在
if not os.path.exists("fig"):
    os.makedirs("fig")

# 特征名称
features = ['科技厅', '排片占比', '场均人次', '排座占比', '黄金占比']

# 重要性得分（你可以在这里修改数值）
feature_importance_svm = [1.2, 4.5, 1.3, 3.0, 0.8]
feature_importance_rf = [1.5, 3.8, 1.7, 2.5, 1.0]

# 画图
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
y_pos = np.arange(len(features))

# SVM 图
axes[0].barh(y_pos, feature_importance_svm)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(features)
axes[0].set_xlabel("重要性")
axes[0].set_title("特征重要性（SVM）")

# 随机森林 图
axes[1].barh(y_pos, feature_importance_rf)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(features)
axes[1].set_xlabel("重要性")
axes[1].set_title("特征重要性（随机森林）")

plt.tight_layout()
plt.savefig("fig/feature_importance.png", dpi=300)  # 以高清晰度保存到 'fig' 文件夹
plt.show()
