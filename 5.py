import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
import lightgbm as lgb

# 设置 Matplotlib 字体，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# **📌 创建文件夹**
fig_dir = "fig_shap"
os.makedirs(fig_dir, exist_ok=True)

# **📌 读取数据**
file_path = "./data/data.csv"
df = pd.read_csv(file_path).dropna()

# **📌 数据预处理**
df["累计票房"] = df["累计票房"].astype(str).str.replace("亿", "").astype(float)
df["人次"] = df["人次"].astype(str).str.replace("万", "").astype(float) * 10000
df["时间"] = df["时间"].astype(int)

# **📌 训练集 (2021-2023) & 测试集 (2024)**
train_df = df[df["时间"].between(2021, 2023)]
test_df = df[df["时间"] == 2024]

# **📌 选择特征**
features = ["票房占比", "排片占比", "场均人次", "排座占比", "黄金占比"]
target = "累计票房"

# **📌 One-Hot 编码**
train_df = pd.get_dummies(train_df, columns=["电影", "版本"], drop_first=True)
test_df = pd.get_dummies(test_df, columns=["电影", "版本"], drop_first=True)

# 保持特征列一致
missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0

features.extend([col for col in train_df.columns if col.startswith("电影_") or col.startswith("版本_")])
train_df = train_df[features + [target]]
test_df = test_df[features + [target]]

# **📌 数据标准化**
X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **📌 MLP 神经网络**
mlp_model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])
mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# **📌 训练 MLP**
mlp_model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, verbose=1, validation_data=(X_test_scaled, y_test))

# **📌 计算 SHAP 值（MLP）**
explainer_mlp = shap.Explainer(mlp_model, X_train_scaled)
shap_values_mlp = explainer_mlp(X_test_scaled)

# **📌 绘制 SHAP Summary Plot (MLP)**
plt.figure(figsize=(12, 8), dpi=300)
shap.summary_plot(shap_values_mlp, X_test_scaled, feature_names=X_train.columns, show=False)
plt.title("SHAP Summary Plot (MLP)")
plt.savefig(os.path.join(fig_dir, "shap_mlp.png"), dpi=300, bbox_inches='tight')
plt.show()

# **📌 训练 XGBoost**
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, eval_metric="rmse")
xgb_model.fit(X_train_scaled, y_train)

# **📌 计算 SHAP 值（XGBoost）**
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_test_scaled)

# **📌 绘制 SHAP Summary Plot (XGBoost)**
plt.figure(figsize=(12, 8), dpi=300)
shap.summary_plot(shap_values_xgb, X_test_scaled, feature_names=X_train.columns, show=False)
plt.title("SHAP Summary Plot (XGBoost)")
plt.savefig(os.path.join(fig_dir, "shap_xgb.png"), dpi=300, bbox_inches='tight')
plt.show()

# **📌 训练 LightGBM**
lgb_model = lgb.LGBMRegressor(n_estimators=100)
lgb_model.fit(X_train_scaled, y_train)

# **📌 计算 SHAP 值（LightGBM）**
explainer_lgb = shap.TreeExplainer(lgb_model)
shap_values_lgb = explainer_lgb.shap_values(X_test_scaled)

# **📌 绘制 SHAP Summary Plot (LightGBM)**
plt.figure(figsize=(12, 8), dpi=300)
shap.summary_plot(shap_values_lgb, X_test_scaled, feature_names=X_train.columns, show=False)
plt.title("SHAP Summary Plot (LightGBM)")
plt.savefig(os.path.join(fig_dir, "shap_lgbm.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ SHAP 分析完成，图片已保存在 {fig_dir}")
