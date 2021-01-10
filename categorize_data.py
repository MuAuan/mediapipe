import pandas as pd
import matplotlib.pyplot as plt
from pandas import plotting
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import csv
import numpy as np

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

df['42']=labels

print(df.head(3))
print(df.iloc[0,:])
header = ['name', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]

with open('./hands/sample_hands9_.csv',  'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(len(df)):
        writer.writerow(np.array(df.iloc[i,:]))
        
df_ = pd.read_csv('./hands/sample_hands9_.csv', sep=',')
print(df_.head(3))
print(df_['42'].astype(int))

plotting.scatter_matrix(df_[df_.columns[1:11]], figsize=(6,6), alpha=0.8, diagonal='kde')
plt.savefig('./hands/scatter_plot_0-10.png')
plt.pause(5)
plt.close()