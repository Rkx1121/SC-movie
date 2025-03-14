

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
# 设置 Matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于 Windows
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# === 1. 读取数据 ===
df = pd.read_csv("box_office_data.csv")  # 读取影厅票房数据

# 转换数据类型
df["累计票房（亿）"] = pd.to_numeric(df["累计票房（亿）"], errors="coerce")

# 电影累计票房排行（所有电影）
plt.figure(figsize=(14, 8), dpi=300)  # 提高分辨率，适合高清保存
df_sorted = df.groupby("电影名")["累计票房（亿）"].sum().sort_values(ascending=False)
sns.barplot(x=df_sorted.values, y=df_sorted.index, palette="crest", edgecolor="black")

# 添加数据标签
for index, value in enumerate(df_sorted.values):
    plt.text(value + 0.1, index, f"{value:.2f}", va='center', fontsize=10)

plt.xlabel("累计票房（亿）", fontsize=14, fontweight='bold')
plt.ylabel("", fontsize=14, fontweight='bold')  # 移除 y 轴标签
plt.title("电影累计票房排行（所有电影）", fontsize=16, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle="--", alpha=0.5)

# 保存高清图片
plt.savefig("电影累计票房排行.png", dpi=300, bbox_inches='tight')
plt.show()
