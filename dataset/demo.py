from torch.utils.data import Dataset, DataLoader
import numpy as np
class MyDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        a = self.data1[index]
        b = self.data2[index]
        return a, b

# 示例数据
data1 = [1, 3, 8, 4, 6]
data2 = [2, 5, 3, 7, 1]

dataset = MyDataset(data1, data2)


dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for e in range(3):
    error = []
    for a, b in dataloader:
        error.append(a-b)
        
    print(max(error))