import pandas as pd
df = pd.read_csv('out.csv')
#把所有的1替换成Yes

df.replace("(847, 686)" ,"(847, 682)", inplace=True)
df.replace("(777, 679)" ,"(777, 684)", inplace=True)
df.replace("(574, 625)" ,"(574, 622)", inplace=True)
df.replace("(816, 63)" ,"(820, 61)", inplace=True)
#保存
df.to_csv('out.csv', index=False)


#two
df.replace("(668, 548)" ,"(670, 548)", inplace=True)
df.replace("(880, 680)" ,"(882, 679)", inplace=True)
df.replace("(402, 1036)" ,"(400, 1037)", inplace=True)
df.replace("(266, 689)" ,"(264, 688)", inplace=True)
df.replace("(261, 288)" ,"(263, 287)", inplace=True)
df.replace("(596, 418)" ,"(599, 417)", inplace=True)
df.replace("(468, 824)" ,"(470, 823)", inplace=True)
df.replace("(592, 91)" ,"(593, 90)", inplace=True)
df.replace("(727, 216)" ,"(730, 215)", inplace=True)
df.replace("(333, 421)" ,"(331, 420)", inplace=True)
