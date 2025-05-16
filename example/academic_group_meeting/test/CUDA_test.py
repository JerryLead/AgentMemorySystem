import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())  # 检查cudnn是否可用
print(torch.backends.cudnn.version())       # 查看cudnn版本
print(torch.tensor([1.0, 2.0]).cuda())
print(torch.tensor([1., 2.], device='cuda:0'))
