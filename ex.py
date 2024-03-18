import torch
t = torch.rand(8,2,6)*10  #[]
print(t)
T = torch.zeros(8)
for i in range(8):
    T[i] =  t[i, :, -1].max()
print(T)


#torch.tensor(1)  # 1