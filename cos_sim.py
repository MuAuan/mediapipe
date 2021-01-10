import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ランダムな10次元のベクトル
a = np.random.rand(41)
b = np.random.rand(41)

print('a: ', a)
print('b: ', b)
print('cos similarity: ', cos_sim(a, b))

df = pd.read_csv('./hands/sample_hands9.csv', sep=',')
#print(df.head(3)) #データの確認
df = df.astype(int)

print(df.iloc[0, :])

for i in range(1,len(df),1):
    for j in range(0,21,2):
        df.iloc[i,2*j+1] = df.iloc[i,2*j+1]-df.iloc[i,1]
        df.iloc[i,2*j] = df.iloc[i,2*j]-df.iloc[i,0]

cs_sim =[]
for i in range(1,len(df),1):
    cs= cos_sim(df.iloc[30,:], df.iloc[i,:])
    #print(df.iloc[i,:]-df.iloc[i,0])
    print('cos similarity: {}-{}'.format(30,i),cs)
    cs_sim.append(cs)
    
plt.figure(figsize=(12, 6))
plt.plot(cs_sim)
plt.ylim(0.9,)
plt.savefig('./hands/cos_sim_hands_plot9.png')
plt.show()
                
