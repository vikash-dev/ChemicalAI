import pandas as pd

#load dataset
df = pd.read_csv('../data/Lipophilicity.csv')

print("dataset load sucessfully ....")
print(df.head())

print("\n Total number of chemicals in list :  ", len(df))