
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置 Seaborn 主题
sns.set_theme(style="whitegrid")

# 解决中文字体问题
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows 中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 读取数据
file_path = "./data/data.csv"
df = pd.read_csv(file_path)

# 清理数据 - 票房转换为浮点数
df["累计票房"] = df["累计票房"].str.replace("亿", "").astype(float)

# **确保时间列为字符串类型，避免排序错误**
df["时间"] = df["时间"].astype(str)

# **按照时间升序排序**
df_sorted = df.sort_values(by="时间")

# 计算每部电影的总票房（所有版本相加），并保持时间顺序
movie_total_box_office = df_sorted.groupby(["电影", "时间"])["累计票房"].sum().reset_index()
movie_total_box_office = movie_total_box_office.sort_values("时间")  # 按时间排序

# 计算每年的总票房
total_box_office = df.groupby("时间")["累计票房"].sum()

# 计算每年的科技厅票房
tech_box_office = df[df["版本"] == "科技厅"].groupby("时间")["累计票房"].sum()

# 计算科技厅票房占比
tech_ratio = tech_box_office / total_box_office

# **创建 fig 目录**
save_path = "./fig"
os.makedirs(save_path, exist_ok=True)  # 确保目录存在

### **第一幅图：电影票房折线图**
plt.figure(figsize=(10, 8))  # 单独创建一张图
ax = sns.lineplot(y=movie_total_box_office["电影"], x=movie_total_box_office["累计票房"], marker="o", linestyle="-", color="b", linewidth=2)

plt.title("21-24年部分电影的票房", fontsize=14, fontweight="bold")
plt.xlabel("累计票房（亿）", fontsize=12)
plt.ylabel("")  # **去掉 Y 轴的“电影”标签**
plt.grid(True, linestyle="--", alpha=0.6)  # 添加网格线

# **方法 1：在折线图上标记时间**
for i, (movie, time, box_office) in enumerate(zip(movie_total_box_office["电影"], movie_total_box_office["时间"], movie_total_box_office["累计票房"])):
    plt.text(box_office, movie, f" {time}", fontsize=10, verticalalignment="center", color="black")  # 右侧标注年份

# # **方法 2：Y 轴上的电影名称后面加时间（替换 y 轴刻度）**
# new_labels = [f"{movie} ({time})" for movie, time in zip(movie_total_box_office["电影"], movie_total_box_office["时间"])]
# ax.set_yticks(movie_total_box_office["电影"])  # 重新设定 y 轴刻度
# ax.set_yticklabels(new_labels, fontsize=10)  # 显示新标签（电影 + 时间）

# **保存第一张图**
save_file1 = os.path.join(save_path, "电影票房折线图.png")
plt.savefig(save_file1, dpi=300, bbox_inches="tight")
plt.show()

### **第二幅图：科技厅票房占比折线图**
plt.figure(figsize=(10, 6))  # 单独创建一张图
sns.lineplot(x=tech_ratio.index.astype(str), y=tech_ratio, marker="o", linestyle="-", color="#1f77b4", linewidth=2)
plt.title("2021-2024 年科技厅票房占比变化", fontsize=14, fontweight="bold")
plt.xlabel("年份", fontsize=12)
plt.ylabel("科技厅票房占比", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)  # 添加网格线

# **保存第二张图**
save_file2 = os.path.join(save_path, "科技厅票房占比折线图.png")
plt.savefig(save_file2, dpi=300, bbox_inches="tight")
plt.show()

print(f"图片已保存到 {save_file1} 和 {save_file2}")

