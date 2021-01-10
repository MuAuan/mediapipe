import os
import pickle
import numpy as np
import PIL.Image
import pandas as pd

df = pd.read_csv('./hands/sample_hands_results_.csv', sep=',')
print(df.head(3))
df = df.astype(int)
print(df['name'], df['42'])

s=len(df) #139 #583
images = []
s1=1
s0 =3 #100
sk=0
for j in range(0,s0,1):
    try:
        for i in range(s1,s,2):
            im = PIL.Image.open('./image/{}'.format(df['42'][i])+'/image{}_'.format(df['42'][i])+str(df['name'][i])+'.png')
            #im = PIL.Image.open('./hands/draw_results/data_plot'+str(i)+'.png')
            im =im.resize(size=(640, 478), resample=PIL.Image.NEAREST)
            images.append(im)
    except Exception as e:
        s1=i+1
        sk += 1
        print(sk,e)
        
print("finish", s, len(images))

images[0].save('./hands/hands_results_.gif', save_all=True, append_images=images[1:s], duration=100*1, loop=0)  