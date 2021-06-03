import pandas as pd
df = pd.read_csv('D:/polynomial/covid19/train_split_v3.txt', sep=" ", names=['file_name','image_name','diseases','Region Name'])
print(df)

#df['man']=pd.Categorical(df["diseases"])
#print(df['man'])
df['dis_cat']=pd.Categorical(df["diseases"]).codes
#print(df['kill'].unique())
#df.to_csv (r'D:/polynomial/covid19/test.csv', index = False, header=True)
df['image_address']='D:/Covid19/data/train/'+df['image_name']
df.to_csv (r'D:/polynomial/covid19/train.csv', index = False, header=True)
#print(x)
