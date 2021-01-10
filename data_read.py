#import mediapipe as mp
#from PIL import Image
import cv2
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.patches as patches

"""
with open('./hands/sample_hands_.csv') as f:
    reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    #for row in reader:
    print(reader)
"""

df = pd.read_csv('./hands/sample_hands_results_.csv', sep=',')
print(df.head(3))
data_num = len(df)
print(data_num)
df = df.astype(int)

#x = pd.DataFrame()
#y = pd.DataFrame()
#x['2']=df['2']-df['0']
#y['3']=df['3']-df['1']
#print(x,y)    
"""
with open('./hands/sample_hands_y.csv',  'w', newline='') as f:
    writer = csv.writer(f)
    x = []
    for j in range(data_num):
        x_ = []
        for i in range(0,21,1):
            x__ = [df['{}'.format(2*i)][j],df['{}'.format(2*i+1)][j]]
            x_.append(x__)
        x.append(x_)
    writer.writerow(np.array(x))
"""

x = []
for j in range(data_num):
    x_ = []
    for i in range(0,21,1):
        x__ = [df['{}'.format(2*i)][j],df['{}'.format(2*i+1)][j]]
        x_.append(x__)
    x.append(x_)

    
y = df['42']    
#for i in range(21):
#print(x[0],y[0])
#print(x[0][10][0],x[0][10][1])
x = np.array(x)
y = np.array(y)
#print(type(x))
print(x.shape,y.shape)

fig = plt.figure()
ax = plt.axes()
while 1:
    for j in range(0,data_num):
        for i in range(20):
            plt.plot(x[j][i][0],x[j][i][1],color='black', marker='o')
        plt.text(600,-120,y[j],size=50)
        plt.xlim(700,0)
        plt.ylim(600,-200)
        plt.title(j)
        plt.pause(0.1)
        plt.savefig('./hands/draw_results/data_plot{}.png'.format(j))
        plt.clf()
    if cv2.waitKey(5) & 0xFF == 27:
        break
        
"""
fig = plt.figure()
ax = plt.axes()
while 1:
    if cv2.waitKey(5) & 0xFF == 27:
        break
    for j in range(390):
        #c = patches.Circle(xy=(df['0'][j],df['1'][j]), radius=0.1, fc='g', ec='r')
        #ax.add_patch(c)
        plt.plot(df['0'][j],df['1'][j],color='black', marker='o')
        plt.plot(df['2'][j],df['3'][j],color='red', marker='o')
        plt.plot(df['4'][j],df['5'][j],color='blue', marker='o')
        plt.plot(df['6'][j],df['7'][j],color='pink', marker='o')
        plt.plot(df['8'][j],df['9'][j],color='purple', marker='o')
        plt.plot(df['10'][j],df['11'][j],color='red', marker='o')
        plt.plot(df['12'][j],df['13'][j],color='blue', marker='o')
        plt.plot(df['14'][j],df['15'][j],color='pink', marker='o')
        plt.plot(df['16'][j],df['17'][j],color='purple', marker='o')
        plt.plot(df['18'][j],df['19'][j],color='red', marker='o')
        plt.plot(df['20'][j],df['21'][j],color='blue', marker='o')
        plt.plot(df['22'][j],df['23'][j],color='pink', marker='o')
        plt.plot(df['24'][j],df['25'][j],color='purple', marker='o')
        plt.plot(df['26'][j],df['27'][j],color='red', marker='o')
        plt.plot(df['28'][j],df['29'][j],color='blue', marker='o')
        plt.plot(df['30'][j],df['31'][j],color='pink', marker='o')
        plt.plot(df['32'][j],df['33'][j],color='purple', marker='o')
        plt.plot(df['34'][j],df['35'][j],color='red', marker='o')
        plt.plot(df['36'][j],df['37'][j],color='blue', marker='o')
        plt.plot(df['38'][j],df['39'][j],color='pink', marker='o')
        plt.plot(df['40'][j],df['41'][j],color='purple', marker='o')
        plt.xlim(0,700)
        plt.ylim(-200,600)
        plt.pause(0.1)
        plt.clf()
    plt.close()
"""
        
