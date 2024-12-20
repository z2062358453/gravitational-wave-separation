# import denoise_pytorch_trainer
# from model.model import Conv_TasNet
# from config import option
# import argparse
# import  torch
# from torch  import nn
# # from model import model
# from config.option import parse
# model_denoise = denoise_pytorch_trainer.MyModel()
# model_path = 'E://Dual-Path-RNN-Pytorch-master//checkpoint_mse//MyModel//best.pt'
# dict = torch.load(model_path, map_location='cuda:0')
# model_denoise.load_state_dict(dict["model_state_dict"])
# model_denoise = model_denoise.cuda()
# yaml_path='./config/Conv_Tasnet/train.yml'
# opt = parse(yaml_path)
# Conv_Tasnet = Conv_TasNet(**opt['Conv_Tasnet'])
# model='./checkpoint/Conv_Tasnet/best.pt'
# dicts = torch.load(model, map_location='cpu')
# Conv_Tasnet.load_state_dict(dicts["model_state_dict"])
# end_to_end_model=nn.Sequential(model_denoise,
#                             Conv_Tasnet)
# end_to_end_model.cuda()
# if __name__ == "__main__":
#     input = torch.rand(3, 32000)
#     # input=  torch.from_numpy(input)
#     input=input.cuda().float()
#     model = end_to_end_model(input)
#     # # print(model)
#     # out = model(input)
#     # print(out)
#
#     k = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print('# of parameters:', k)
import denoise_pytorch_trainer
from model.model import Conv_TasNet
from config import option
import argparse
import torch
from torch import nn
from config.option import parse
import numpy as np
from model.model_rnn import Dual_RNN_model
# 加载第一个模型
model_denoise = denoise_pytorch_trainer.MyModel()
model_path = 'E://Dual-Path-RNN-Pytorch-master//checkpoint_mse//MyModel//best.pt'
dict1 = torch.load(model_path, map_location='cuda:0')
model_denoise.load_state_dict(dict1["model_state_dict"])
model_denoise = model_denoise.cuda()

# 加载第二个模型
yaml_path = './config/Conv_Tasnet/train.yml'
opt = parse(yaml_path)
Conv_Tasnet = Conv_TasNet(**opt['Conv_Tasnet'])
model_path = './checkpoint/Conv_Tasnet/best.pt'
dict2 = torch.load(model_path, map_location='cpu')
Conv_Tasnet.load_state_dict(dict2["model_state_dict"])
Conv_Tasnet = Conv_Tasnet.cuda()
yaml_path1 = './config/Dual_RNN/train_rnn.yml'
opt = parse(yaml_path1)
Dual_Path_RNN= Dual_RNN_model(**opt['Dual_Path_RNN'])
model_path = './checkpoint/Dual_Path_RNN/best.pt'
dict3 = torch.load(Dual_Path_RNN, map_location='cpu')
Dual_Path_RNN.load_state_dict(dict3["model_state_dict"])
Dual_Path_RNN = Dual_Path_RNN.cuda()
# 将两个模型组合成一个模型
class CombinedModel(nn.Module):
    def __init__(self, model_denoise, Dual_Path_RNN):
        super(CombinedModel, self).__init__()
        self.model_denoise = model_denoise
        self.Dual_Path_RNN = Dual_Path_RNN

    def forward(self, x):
        # 对第一个模型进行前向传播
        x = self.model_denoise(x)
        # 对第二个模型进行前向传播
        x = self.Conv_Tasnet(x)
        return x

# 创建组合模型对象
end_to_end_model = CombinedModel(model_denoise, Dual_Path_RNN)
end_to_end_model = end_to_end_model.cuda()

# # 在组合模型中计算参数数量
# num_params = sum(p.numel() for p in end_to_end_model.parameters() if p.requires_grad)
# print('# of parameters:', num_params)
if __name__ == "__main__":
    input = torch.rand(16, 8192)
    # input=  torch.from_numpy(input)
    input=input.cuda().float()
    model = CombinedModel(model_denoise, Dual_Path_RNN)
    model =model.cuda()
    print(model)
    out = model(input)
    print(out[0].shape)

    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of parameters:', k)