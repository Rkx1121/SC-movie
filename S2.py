# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.metrics import mean_squared_error
#
# # 设置 Matplotlib 中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于 Windows
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
# # === 1. 读取数据 ===
# data = pd.read_csv("box_office_data.csv")  # 读取影厅票房数据
#
# # === 2. 数据预处理 ===
# # 转换百分比数据
# percent_cols = ['票房占比', '排片占比', '排座占比', '黄金占比', '上座率']
# for col in percent_cols:
#     data[col] = data[col].str.rstrip('%').astype(float) / 100  # 转换百分比为小数
#
# # 影厅版本编码
# label_encoder = LabelEncoder()
# data['版本编码'] = label_encoder.fit_transform(data['版本'])  # 转换为数字
#
# # 选择特征
# X = data[['版本编码', '排片占比', '场均人次', '排座占比', '黄金占比']]
# y = data['累计票房（亿）']
#
# # 归一化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
# # 创建输出文件夹
# output_dir = "fig"
# os.makedirs(output_dir, exist_ok=True)
#
# # 创建文本输出文件
# text_file = "output.txt"
# with open(text_file, "w", encoding="utf-8") as f:
#     # === 3. 线性回归分析 ===
#     lr_model = LinearRegression()
#     lr_model.fit(X_train, y_train)
#     y_pred_lr = lr_model.predict(X_test)
#     lr_score = lr_model.score(X_test, y_test)
#     lr_mse = mean_squared_error(y_test, y_pred_lr)  # 计算MSE
#     print("线性回归 R² Score:", lr_score)
#     print("线性回归 MSE:", lr_mse)
#
#     f.write(f"线性回归 R² Score: {lr_score:.4f}\n")
#     f.write(f"线性回归 MSE: {lr_mse:.4f}\n")
#
#     # === 4. 随机森林回归 ===
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_model.fit(X_train, y_train)
#     y_pred_rf = rf_model.predict(X_test)
#     rf_score = rf_model.score(X_test, y_test)
#     rf_mse = mean_squared_error(y_test, y_pred_rf)  # 计算MSE
#     print("随机森林 R² Score:", rf_score)
#     print("随机森林 MSE:", rf_mse)
#
#     f.write(f"随机森林 MSE: {rf_mse:.4f}\n")
#     f.write(f"随机森林 R² Score: {rf_score:.4f}\n")
#
#     # === 5. XGBoost 回归 ===
#     xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
#     xgb_model.fit(X_train, y_train)
#     y_pred_xgb = xgb_model.predict(X_test)
#     xgb_score = xgb_model.score(X_test, y_test)
#     xgb_mse = mean_squared_error(y_test, y_pred_xgb)  # 计算MSE
#     print("XGBoost MSE:", xgb_mse)
#     print("XGBoost R² Score:", xgb_score)
#
#     f.write(f"XGBoost MSE: {xgb_mse:.4f}\n")
#     f.write(f"XGBoost R² Score: {xgb_score:.4f}\n")
#
#     # === 6. 构建神经网络 ===
#     nn_model = Sequential([
#         Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#         Dense(32, activation='relu'),
#         Dense(1)  # 预测票房
#     ])
#
#     nn_model.compile(optimizer='adam', loss='mse')
#     history = nn_model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, validation_data=(X_test, y_test))
#
#     # 训练损失曲线
#     plt.figure(figsize=(8, 4))
#     plt.plot(history.history['loss'], label='训练损失')
#     plt.plot(history.history['val_loss'], label='验证损失')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('神经网络训练损失曲线')
#     loss_plot_path = os.path.join(output_dir, "nn_loss.png")
#     plt.savefig(loss_plot_path, dpi=300)
#     plt.close()
#     f.write(f"神经网络训练损失曲线图: {loss_plot_path}\n")
#
#     # === 7. 可视化分析 ===
#     plt.figure(figsize=(10, 5))
#     plt.scatter(y_test, y_pred_lr, label="线性回归", alpha=0.6)
#     plt.scatter(y_test, y_pred_rf, label="随机森林", alpha=0.6)
#     plt.scatter(y_test, y_pred_xgb, label="XGBoost", alpha=0.6)
#     plt.xlabel("真实票房（亿）")
#     plt.ylabel("预测票房（亿）")
#     plt.legend()
#     plt.title("不同模型的票房预测结果")
#     scatter_plot_path = os.path.join(output_dir, "model_comparison.png")
#     plt.savefig(scatter_plot_path, dpi=300)
#     plt.close()
#     f.write(f"模型比较散点图: {scatter_plot_path}\n")
#
#     # 计算特征重要性（随机森林）
#     feature_importance = rf_model.feature_importances_
#     feature_names = X.columns
#
#     plt.figure(figsize=(8, 4))
#     sns.barplot(x=feature_importance, y=feature_names)
#     plt.title("特征重要性（随机森林）")
#     plt.xlabel("重要性")
#     plt.ylabel("特征")
#     feature_plot_path = os.path.join(output_dir, "feature_importance.png")
#     plt.savefig(feature_plot_path, dpi=300)
#     plt.close()
#     f.write(f"特征重要性图: {feature_plot_path}\n")
#



import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

# 设置 Matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于 Windows
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# === 1. 读取数据 ===
data = pd.read_csv("box_office_data.csv")  # 读取影厅票房数据

# === 2. 数据预处理 ===
# 转换百分比数据
percent_cols = ['票房占比', '排片占比', '排座占比', '黄金占比', '上座率']
for col in percent_cols:
    data[col] = data[col].str.rstrip('%').astype(float) / 100  # 转换百分比为小数

# 影厅版本编码
label_encoder = LabelEncoder()
data['版本编码'] = label_encoder.fit_transform(data['版本'])  # 转换为数字

# 选择特征
X = data[['版本编码', '排片占比', '场均人次', '排座占比', '黄金占比']]
y = data['累计票房（亿）']

# 归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建输出文件夹
output_dir = "fig"
os.makedirs(output_dir, exist_ok=True)

# 创建文本输出文件
text_file = "output.txt"
with open(text_file, "w", encoding="utf-8") as f:
    # === 3. 线性回归分析 ===
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    lr_score = lr_model.score(X_test, y_test)
    lr_mse = mean_squared_error(y_test, y_pred_lr)  # 计算MSE
    print("线性回归 R² Score:", lr_score)
    print("线性回归 MSE:", lr_mse)

    f.write(f"线性回归 R² Score: {lr_score:.4f}\n")
    f.write(f"线性回归 MSE: {lr_mse:.4f}\n")

    # === 4. 随机森林回归 ===
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_score = rf_model.score(X_test, y_test)
    rf_mse = mean_squared_error(y_test, y_pred_rf)  # 计算MSE
    print("随机森林 R² Score:", rf_score)
    print("随机森林 MSE:", rf_mse)

    f.write(f"随机森林 MSE: {rf_mse:.4f}\n")
    f.write(f"随机森林 R² Score: {rf_score:.4f}\n")

    # === 5. XGBoost 回归 ===
    # xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    # xgb_model.fit(X_train, y_train)
    # y_pred_xgb = xgb_model.predict(X_test)
    # xgb_score = xgb_model.score(X_test, y_test)
    # xgb_mse = mean_squared_error(y_test, y_pred_xgb)  # 计算MSE
    # print("XGBoost MSE:", xgb_mse)
    # print("XGBoost R² Score:", xgb_score)
    #
    # f.write(f"XGBoost MSE: {xgb_mse:.4f}\n")
    # f.write(f"XGBoost R² Score: {xgb_score:.4f}\n")
    xgb_model = XGBRegressor(
        n_estimators=500,  # 增加树的数量
        learning_rate=0.05,  # 调整学习率
        max_depth=6,  # 设置树的深度
        subsample=0.8,  # 设置训练数据的随机抽样比例
        colsample_bytree=0.8,  # 设置每棵树的特征随机抽样比例
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_score = xgb_model.score(X_test, y_test)
    xgb_mse = mean_squared_error(y_test, y_pred_xgb)  # 计算MSE
    print("XGBoost MSE:", xgb_mse)
    print("XGBoost R² Score:", xgb_score)

    f.write(f"XGBoost MSE: {xgb_mse:.4f}\n")
    f.write(f"XGBoost R² Score: {xgb_score:.4f}\n")

    # 打印 XGBoost 模型的特征重要性
    feature_importance = xgb_model.feature_importances_
    feature_names = X.columns
    f.write("XGBoost 特征重要性:\n")
    for name, importance in zip(feature_names, feature_importance):
        f.write(f"{name}: {importance:.4f}\n")

    # === 6. 岭回归分析 ===
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    ridge_score = ridge_model.score(X_test, y_test)
    ridge_mse = mean_squared_error(y_test, y_pred_ridge)  # 计算MSE
    print("岭回归 R² Score:", ridge_score)
    print("岭回归 MSE:", ridge_mse)

    f.write(f"岭回归 R² Score: {ridge_score:.4f}\n")
    f.write(f"岭回归 MSE: {ridge_mse:.4f}\n")

    # === 7. 支持向量机回归 ===
    svm_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    svm_mse = mean_squared_error(y_test, y_pred_svm)
    svm_score = svm_model.score(X_test, y_test)
    print("支持向量机回归 MSE:", svm_mse)
    print("支持向量机回归 R² Score:", svm_score)

    f.write(f"支持向量机回归 MSE: {svm_mse:.4f}\n")
    f.write(f"支持向量机回归 R² Score: {svm_score:.4f}\n")


    # === 8. 构建神经网络 ===
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)  # 预测票房
    ])
    # nn_model = Sequential([
    #     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # 增加层数和神经元
    #     Dropout(0.3),  # 增加Dropout率
    #     Dense(64, activation='relu'),
    #     Dropout(0.3),
    #     Dense(32, activation='relu'),
    #     Dropout(0.2),
    #     Dense(1)  # 预测票房
    # ])
    # nn_model = Sequential([
    #     Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),  # 增加神经元并添加L2正则化
    #     Dropout(0.4),  # 增加Dropout比例
    #     Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    #     Dropout(0.4),
    #     Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    #     Dropout(0.3),
    #     Dense(32, activation='relu'),
    #     Dropout(0.2),
    #     Dense(1)  # 预测票房
    # ])
    optimizer = Adam(learning_rate=0.001)  # 设置学习率
    nn_model.compile(optimizer=optimizer, loss='mse')
    # 设置EarlyStopping，防止过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = nn_model.fit(
        X_train, y_train, epochs=200, batch_size=8, verbose=1,
        validation_data=(X_test, y_test), callbacks=[early_stopping]
    )
    # history = nn_model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, validation_data=(X_test, y_test))

    # 训练损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('神经网络训练损失曲线')
    loss_plot_path = os.path.join(output_dir, "nn_loss.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    f.write(f"神经网络训练损失曲线图: {loss_plot_path}\n")

    # === 7. 可视化分析 ===
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred_lr, label="线性回归", alpha=0.5)
    plt.scatter(y_test, y_pred_rf, label="随机森林", alpha=0.5)
    plt.scatter(y_test, y_pred_xgb, label="XGBoost", alpha=0.5)
    plt.xlabel("真实票房（亿）")
    plt.ylabel("预测票房（亿）")
    plt.legend()
    plt.title("不同模型的票房预测结果")
    scatter_plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(scatter_plot_path, dpi=300)
    plt.close()
    f.write(f"模型比较散点图: {scatter_plot_path}\n")

    # 计算特征重要性（随机森林）
    rf_importance = rf_model.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(8, 4))
    sns.barplot(x=rf_importance, y=feature_names)
    plt.title("特征重要性（随机森林）")
    plt.xlabel("重要性")
    plt.ylabel("特征")
    feature_plot_path = os.path.join(output_dir, "rf_importance.png")
    plt.savefig(feature_plot_path, dpi=300)
    plt.close()
    f.write(f"特征重要性图: {feature_plot_path}\n")

    # 计算特征重要性（岭回归）
    ridge_importance = ridge_model.coef_
    feature_names = X.columns

    plt.figure(figsize=(8, 4))
    sns.barplot(x=ridge_importance, y=feature_names)
    plt.title("特征重要性（岭回归）")
    plt.xlabel("重要性")
    plt.ylabel("特征")
    feature_plot_path = os.path.join(output_dir, "ridge_importance.png")
    plt.savefig(feature_plot_path, dpi=300)
    plt.close()
    f.write(f"特征重要性图: {feature_plot_path}\n")

    # 计算特征重要性（SVM）
    explainer = shap.Explainer(svm_model.predict, X_train)
    shap_values = explainer(X_test)
    # 计算特征重要性（取 SHAP 绝对值的均值）
    SVM_importance = np.abs(shap_values.values).mean(axis=0)
    feature_names = X.columns
    plt.figure(figsize=(8, 4))
    sns.barplot(x=ridge_importance, y=feature_names)
    plt.title("特征重要性（支持向量机）")
    plt.xlabel("重要性")
    plt.ylabel("特征")
    feature_plot_path = os.path.join(output_dir, "SVM_importance.png")
    plt.savefig(feature_plot_path, dpi=300)
    plt.close()
    f.write(f"特征重要性图: {feature_plot_path}\n")

    # 计算特征重要性（线性回归）
    lr_importance = np.abs(lr_model.coef_)
    feature_names = X.columns
    plt.figure(figsize=(8, 4))
    sns.barplot(x=ridge_importance, y=feature_names)
    plt.title("特征重要性（线性回归）")
    plt.xlabel("重要性")
    plt.ylabel("特征")
    feature_plot_path = os.path.join(output_dir, "lr_importance.png")
    plt.savefig(feature_plot_path, dpi=300)
    plt.close()
    f.write(f"特征重要性图: {feature_plot_path}\n")

    # === 8. 计算相关系数 ===
    correlation_matrix = data[['累计票房（亿）', '版本编码', '排片占比', '场均人次', '排座占比', '黄金占比']].corr()

    # 打印相关系数矩阵
    print("相关系数矩阵:")
    print(correlation_matrix)
    f.write("相关系数矩阵:\n")
    f.write(correlation_matrix.to_string())
    f.write("\n")

    # 可视化相关系数矩阵
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('票房与特征的相关系数')
    correlation_plot_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(correlation_plot_path, dpi=300)
    plt.close()
    f.write(f"相关系数矩阵图: {correlation_plot_path}\n")

    # === 9. 混淆矩阵 ===
    # 对票房进行二值化处理，分为高票房和低票房（假设票房中位数为阈值）
    median_ticket_sales = data['累计票房（亿）'].median()
    y_bin = (y > median_ticket_sales).astype(int)  # 票房大于中位数为1，否则为0

    # 划分训练集和测试集的二值标签
    y_train_bin, y_test_bin = train_test_split(y_bin, test_size=0.2, random_state=42)

    # 生成混淆矩阵
    y_pred_bin = lr_model.predict(X_test)
    y_pred_bin = (y_pred_bin > median_ticket_sales).astype(int)  # 预测的票房二值化

    cm = confusion_matrix(y_test_bin, y_pred_bin)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["低票房", "高票房"])

    # 显示混淆矩阵
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues')
    plt.title("混淆矩阵（线性回归）")
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    f.write(f"混淆矩阵图: {cm_plot_path}\n")

    f.write("\\end{document}")
