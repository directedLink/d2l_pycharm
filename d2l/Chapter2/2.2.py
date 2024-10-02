import os

import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd

data = pd.read_csv(data_file)
print(data)

inputs1, inputs2, outputs = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
inputs1 = inputs1.fillna(inputs1.mean())
inputs = pd.concat([inputs1, inputs2], axis=1)
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

X = torch.tensor(inputs.to_numpy(float))
y = torch.tensor(outputs.to_numpy(float))

print(X, y)
