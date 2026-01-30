import pandas as pd
import numpy as np
import os

data_dir = '../data'

df = pd.read_csv(os.path.join(data_dir, 'names_labels_train.csv'))
print("CSV Header ....")
print(df.head())


data = np.load(os.path.join(data_dir, 'names_onehots.npy'), allow_pickle = True)
if data.shape == ():
    unpacked_data = data.item()
    print ("----------unpacked data type --------")
    print(type(unpacked_data))

     # If it's a dictionary, let's see the keys
    if isinstance(unpacked_data, dict):
        print("Keys found:", unpacked_data.keys())
    else:
        print("Content preview:", str(unpacked_data)[:500])

else:
    print("--- Features Shape ---")
    print(data.shape)
    print(data[:5]) 

