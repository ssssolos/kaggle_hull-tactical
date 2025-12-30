import re
import os
import pandas as pd
import polars as pl
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==========================
# CONFIGURATION
# ==========================
INPUT_PATH = '/kaggle/input/hull-tactical-market-prediction/'
MODEL_PATH = '/tmp/lgbm_model.pkl'
TARGET_COL = 'market_forward_excess_returns'

# 指定使用的特征
#SELECTED_FEATURES = ['M4', 'E19', 'P3', 'M3', 'P7', 'P4', 'S2', 'S5', 'V13', 'P6', 
#                     'P12', 'P13', 'P5', 'V3', 'V5', 'V7', 'S6', 'M2', 'M8', 'S9']
EXCLUDE_COLS = ['date_id', 'forward_returns', 'risk_free_rate', TARGET_COL]

# ==========================
# 1. LOAD DATA
# ==========================
print("Loading data...")
train = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
print(f"Train shape: {train.shape}")

# ==========================
# 2. PREPROCESSING
# ==========================


def handle_missing(train):
    """处理缺失值"""
    def get_feature_group(prefix):
        return [c for c in train.columns if re.match(f'^{prefix}[0-9]+$', c)]
    groups = {
        'D': get_feature_group('D'),
        'E': get_feature_group('E'),
        'I': get_feature_group('I'),
        'M': get_feature_group('M'),
        'P': get_feature_group('P'),
        'S': get_feature_group('S'),
        'V': get_feature_group('V'),
    }
    # 按组进行缺失处理
    for gname, cols in groups.items():
        if not cols:
            continue
        if gname == 'D':
            train[cols] = train[cols].fillna(0)
        elif gname in ['E', 'I']:
            train[cols] = train[cols].ffill().fillna(train[cols].median())
        elif gname in ['M', 'P', 'V']:
            train[cols] = train[cols].ffill().fillna(train[cols].mean())
        elif gname == 'S':
            for c in cols:
                col = train[c].copy()
                col = col.fillna(train[c].rolling(5, min_periods=1).mean())
                col = col.ffill(limit=5)
                col = col.ffill().fillna(train[c].mean())
                train[c] = col
    return train


def handle_outliers(df, clip=1):
    """处理异常值"""
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c not in EXCLUDE_COLS]
    for col in numeric_cols:
        low = df[col].quantile(clip/100)
        high = df[col].quantile(1-clip/100)
        df[col] = df[col].clip(low, high)
    return df

print("Preprocessing training data...")
train = train.iloc[1005:].reset_index(drop=True)
drop_cols = ['forward_returns', 'risk_free_rate']
drop_cols = [c for c in drop_cols if c in train.columns]

# 缺失率过高的列
missing_ratio = train.isna().mean()
too_missing = missing_ratio[missing_ratio > 0.3].index.tolist()
len(too_missing)

train.drop(columns=drop_cols+too_missing, inplace=True, errors='ignore')

train = handle_missing(train)
train = handle_outliers(train)

print(f"Train shape: {train.shape}")
# ==========================
# 3. FEATURE SELECTION
# ==========================
def select_top_features(train, target=TARGET_COL, top_k=20):
    """
    使用 LightGBM 训练模型并返回 top K 特征
    """
    X = train.drop(columns=[target])
    y = train[target]

    # 训练 LightGBM 模型
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
    model.fit(X, y)

    # 提取特征重要性
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    top_features = imp.head(top_k)["feature"].tolist()
    print(top_features)
    return top_features
SELECTED_FEATURES = select_top_features(train)
FEATURES = [f for f in SELECTED_FEATURES if f in train.columns]
missing_features = set(SELECTED_FEATURES) - set(FEATURES)

if missing_features:
    print(f"⚠️ Warning: Missing features: {missing_features}")
    
print(f"✓ Using {len(FEATURES)} selected features")

# ==========================
# 4. MODEL TRAINING
# ==========================
X_train = train[FEATURES]
y_train = train[TARGET_COL]

lgbm = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    n_estimators=2000,
    learning_rate=0.08,
    num_leaves=63,
    max_depth=8,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

print("\n" + "="*50)
print("Training LightGBM model...")
print("="*50)
lgbm.fit(X_train, y_train)
print("✓ Training complete.")

# 保存模型
model_data = {
    'model': lgbm,
    'features': FEATURES
}
joblib.dump(model_data, MODEL_PATH)
print(f"✓ Model saved to {MODEL_PATH}")

# 显示特征重要性
fi = pd.DataFrame({
    'feature': FEATURES, 
    'importance': lgbm.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(fi.head(10).to_string(index=False))

# ==========================
# 5. PREDICTION FUNCTION
# ==========================
def predict(test_df_pl: pl.DataFrame) -> pd.DataFrame:
    """
    预测函数 - Kaggle评测API接口
    输入: Polars DataFrame
    输出: Pandas DataFrame with columns ['date_id', 'prediction']
    """
    # 转换为pandas
    test_df_pd = test_df_pl.to_pandas()
    
    # 加载模型
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    features = model_data['features']
    
    # 预处理测试数据
    test_df_pd = handle_missing(test_df_pd)
    test_df_pd = handle_outliers(test_df_pd)
    
    # 提取特征
    X_test = test_df_pd[features]
    
    # 预测超额收益
    predictions = model.predict(X_test)
    
    # 转换为交易信号 (0: 不持仓, 1: 部分持仓, 2: 全仓)
    # 策略: 负收益->0, 低正收益->1, 高正收益->2
    q9 = np.quantile(train[TARGET_COL], 0.9)
    signals = np.where(
        predictions <= 0, 0,
        np.where(predictions <= q9, 1, 2)
    )
    
    # 调试输出
    print(f"Predictions stats: min={predictions.min():.6f}, max={predictions.max():.6f}, mean={predictions.mean():.6f}")
    print(f"Signal distribution: {pd.Series(signals).value_counts().sort_index().to_dict()}")
    
    # 返回结果
    return pd.DataFrame({
        'date_id': test_df_pd['date_id'],
        'prediction': signals
    })

# ==========================
# 6. INFERENCE SERVER
# ==========================
print("\n" + "="*50)
print("Setting up inference server...")
print("="*50)

from kaggle_evaluation.default_inference_server import DefaultInferenceServer

inference_server = DefaultInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    print("🚀 Serving predictions for the competition...")
    inference_server.serve()
else:
    print("🧪 Running local gateway for testing...")
    inference_server.run_local_gateway((INPUT_PATH,))

print("\n✓ Submission script finished.")