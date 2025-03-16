import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb

# 设置 Seaborn 风格，提高可视化美观度
sns.set_theme(style="whitegrid")

# **创建保存图片的文件夹**
fig_dir = "fig"
os.makedirs(fig_dir, exist_ok=True)

# **读取数据**
file_path = "./data/data.csv"
df = pd.read_csv(file_path)

# 数据预处理
df = df.dropna()
df["累计票房"] = df["累计票房"].astype(str).str.replace("亿", "").astype(float)
df["人次"] = df["人次"].astype(str).str.replace("万", "").astype(float) * 10000
df["时间"] = df["时间"].astype(int)

# 训练集: 2021-2023, 测试集: 2024
train_df = df[df["时间"].between(2021, 2023)]
test_df = df[df["时间"] == 2024]

# **特征选择**
features = ["票房占比", "排片占比", "场均人次", "排座占比", "黄金占比"]
target = "累计票房"

# **使用 pd.get_dummies 进行 One-Hot 编码**
train_df = pd.get_dummies(train_df, columns=["电影", "版本"], drop_first=True)
test_df = pd.get_dummies(test_df, columns=["电影", "版本"], drop_first=True)

# **保持特征列一致**
missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0

features.extend([col for col in train_df.columns if col.startswith("电影_") or col.startswith("版本_")])
train_df = train_df[features + [target]]
test_df = test_df[features + [target]]

# **数据标准化**
X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **构建 MLP 模型**
mlp_model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])
mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# **训练 MLP 并记录过程**
history = mlp_model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=16, verbose=1)

# **📌 绘制并保存 训练过程曲线**
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# 损失曲线
axes[0].plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle="--")
axes[0].set_xlabel('Epochs', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('MLP Training Loss Curve', fontsize=14, fontweight='bold')
axes[0].legend()

# MAE 曲线
axes[1].plot(history.history['mae'], label='Train MAE', color='#2ca02c', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Validation MAE', color='#d62728', linewidth=2, linestyle="--")
axes[1].set_xlabel('Epochs', fontsize=12)
axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
axes[1].set_title('MLP Training MAE Curve', fontsize=14, fontweight='bold')
axes[1].legend()

# **保存高清图片**
training_curve_path = os.path.join(fig_dir, "training_curves.png")
plt.savefig(training_curve_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ 训练曲线已保存: {training_curve_path}")

# **MLP 预测**
y_pred_mlp = mlp_model.predict(X_test_scaled)

# **计算 MSE 和 R²**
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)
print(f"MLP - MSE: {mse_mlp:.4f}, R²: {r2_mlp:.4f}")

# **XGBoost 训练**
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
xgb_model.fit(X_train_scaled, y_train)

# **XGBoost 预测**
y_pred_xgb = xgb_model.predict(X_test_scaled)

# **计算 MSE 和 R²**
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost - MSE: {mse_xgb:.4f}, R²: {r2_xgb:.4f}")

# **📌 绘制并保存 XGBoost 特征重要性**
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
xgb.plot_importance(xgb_model, importance_type="gain", ax=ax, title="Feature Importance (XGBoost)")
plt.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ 特征重要性图已保存: {os.path.join(fig_dir, 'feature_importance.png')}")




