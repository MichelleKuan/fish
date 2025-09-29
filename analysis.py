import os
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import sparse
from scipy.sparse.linalg import spsolve

# 1. 資料夾路徑
data_folder = r"D:\NTUST\Experiment data\250716\average"

# 定義分類規則
def get_label_from_filename(filename):
    name = filename.lower()
    if "viscose" in name or "rayon" in name or "tencel" in name:
        return "MMCF"
    elif "polyester" in name:
        return "PES"
    elif "cotton" in name:
        return "Cotton"
    elif "nylon" in name:
        return "Nylon"
    elif "linene" in name:
        return "Linene"
    else:
        return "Unknown"

# 3. 讀取檔案（支援 txt）
spectra = []
labels = []
file_list = []

print("資料夾內容:", os.listdir(data_folder))

for file in os.listdir(data_folder):
    if file.endswith(".txt") or file.endswith(".csv"):
        filepath = os.path.join(data_folder, file)

        # 嘗試讀檔（判斷分隔符號）
        try:
            df = pd.read_csv(filepath, sep=None, engine='python')  # 自動偵測分隔符號
        except Exception:
            df = pd.read_csv(filepath, sep="\t")  # 預設 Tab 分隔

        print(f"處理檔案: {file}, shape={df.shape}")

        if df.shape[1] >= 2:
            intensity = df.iloc[:, 1].values  # 第二欄是強度
            spectra.append(intensity)
            labels.append(get_label_from_filename(file))
            file_list.append(file)
        else:
            print(f"⚠️ {file} 欄位數不足，無法使用")

# 檢查是否讀到資料
if len(spectra) == 0:
    raise ValueError("❌ 沒有讀到任何光譜，請檢查檔案內容")

spectra = np.array(spectra)
print(f"✅ 成功讀取 {len(spectra)} 筆光譜，每筆長度 {spectra.shape[1]}")
print("分類統計:", pd.Series(labels).value_counts())

# 使用 StandardScaler 做標準化
scaler = StandardScaler()
spectra_scaled = scaler.fit_transform(spectra)

# 印出標準化後的資料
df_scaled = pd.DataFrame(
    spectra_scaled,
    index=file_list,
    columns=[f"feature_{i+1}" for i in range(spectra_scaled.shape[1])],
)
print("標準化後的光譜資料（前五筆）:")
print(df_scaled.head())

# 2. PCA 降維
pca = PCA(n_components=20)
spectra_pc1 = pca.fit_transform(spectra_scaled)
explained_ratio = pca.explained_variance_ratio_[0]

df_pc1 = pd.DataFrame({
    'Index': range(len(file_list)),
    'PC1': spectra_pc1[:, 0],
    'Filename': file_list,
    'Label': labels
})

fig = px.scatter(
    df_pc1,
    x='Index',
    y='PC1',
    color='Label',
    hover_name='Filename',
    title=f"PCA Projection (1D) — Explained Variance: {explained_ratio:.2%}",
    width=900,
    height=500
)
fig.update_traces(marker=dict(size=6, opacity=0.7))
fig.show()
