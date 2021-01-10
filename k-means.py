import pandas as pd
import matplotlib.pyplot as plt
from pandas import plotting
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv('./hands/sample_hands9.csv', sep=',')
print(df.head(3))
df = df.astype(int)

plotting.scatter_matrix(df[df.columns[1:11]], figsize=(6,6), alpha=0.8, diagonal='kde')
plt.savefig('./hands/scatter_plot0-10.png')
plt.pause(5)
plt.close()

# この例では 3 つのグループに分割 (メルセンヌツイスターの乱数の種を 10 とする)
kmeans_model = KMeans(n_clusters=3, random_state=10).fit(df.iloc[:, :])
# 分類結果のラベルを取得する
labels = kmeans_model.labels_

# 分類結果を確認
print(len(labels),labels)

# それぞれに与える色を決める。
color_codes = {0:'#00FF00', 1:'#FF0000', 2:'#0000FF'} #,3:'#FF00FF', 4:'#00FFFF', 5:'#FFFF00', 6:'#000000'}
# サンプル毎に色を与える。
colors = [color_codes[x] for x in labels]

# 色分けした Scatter Matrix を描く。
plotting.scatter_matrix(df[df.columns[1:11]], figsize=(6,6),c=colors, diagonal='kde', alpha=0.8)   #データのプロット
plt.savefig('./hands/scatter_color_plot0-10.png')
plt.pause(1)
plt.close()

#主成分分析の実行
pca = PCA()
pca.fit(df.iloc[:, :])
PCA(copy=True, n_components=None, whiten=False)

# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(df.iloc[:, :])

# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(6, 6))
for x, y, name in zip(feature[:, 0], feature[:, 1], df.iloc[:, 0]):
    plt.text(x, y, name, alpha=0.8, size=10)
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, color=colors[:])
plt.title("Principal Component Analysis")
plt.xlabel("The first principal component score")
plt.ylabel("The second principal component score")
plt.savefig('./hands/PCA_hands_plot.png')
plt.pause(1)
plt.close()
