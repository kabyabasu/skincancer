import torch

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

list1 = [1,2,3,4,5,6]

tourch_list = torch.tensor(list1).to(device)

print(tourch_list)
print(tourch_list.device)