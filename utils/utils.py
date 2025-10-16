import torch
import os
import random
import pickle
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

def get_tile_map(args, if_load = True):
    path = f"./data"
    if not os.path.exists(path): os.makedirs(path)
    path += f"/{args.dataset}.pkl"
    if if_load and os.path.exists(path):
        with open(path, "rb") as tf:
            tile_map = pickle.load(tf)

    else:
        data_dir = f'{args.inst_path}/{args.dataset}_ms/tumor/5x'
        tile_map = {} # key:slideID
        for i in os.listdir(data_dir): # make tile ID lists for all slides
            tiles = os.listdir(f'{data_dir}/{i}')
            tiles.sort(key=lambda x: int(x.split('.')[-2].split('/')[-1]))
            tile_map[i] = tiles

        with open(path, "wb") as tf:
            pickle.dump(tile_map,tf)
    return tile_map

def build_inst_multi(dic, sample_list, args, prediction=False):
    all_inst = []
    all_num = [0,0]
    tile_dir_5x = f'{args.inst_path}/{args.dataset}_ms/tumor/5x'
    tile_dir_10x = f'{args.inst_path}/{args.dataset}_ms/tumor/10x'
    tile_dir_20x = f'{args.inst_path}/{args.dataset}_ms/tumor/20x'

    for data in sample_list:
        slideID = data[0]
        class_label = data[-1]
        if class_label == 2 and not prediction: continue
        if slideID not in dic.keys(): continue

        inst_list = []
        for id, tile_list in dic[slideID]:
            for i in range(len(tile_list)):
                tile_i = tile_list[i]
                inst_5x = f'{tile_dir_5x}/{id}/{tile_i}'
                inst_10x = f'{tile_dir_10x}/{id}/{tile_i}'
                inst_20x = f'{tile_dir_20x}/{id}/{tile_i}'
                inst_list.append([inst_5x, inst_10x, inst_20x, class_label, slideID])

        inst_num = len(inst_list)           
            
        all_inst += inst_list[:inst_num]
        if not prediction: all_num[class_label] += inst_num 

    if not prediction:
        low_num, high_num = all_num
        print(f'high risk ratio = {high_num / (low_num + high_num)}', low_num, high_num)
        weights = torch.FloatTensor([high_num / low_num, 1])
    else:
        return all_inst
    return all_inst, weights

def data_load_multi(data, device):
    tensor_x1, tensor_x2, tensor_x3, tensor_y = [], [], [], []
    for i in data:
        tensor_x1.append(torchvision.io.decode_jpeg(torchvision.io.read_file(i[0]), device=device).float().div_(255))
        tensor_x2.append(torchvision.io.decode_jpeg(torchvision.io.read_file(i[1]), device=device).float().div_(255))
        tensor_x3.append(torchvision.io.decode_jpeg(torchvision.io.read_file(i[2]), device=device).float().div_(255))
        tensor_y.append(i[3])
    tensor_x1 = torch.stack(tensor_x1, dim=0)
    tensor_x2 = torch.stack(tensor_x2, dim=0)
    tensor_x3 = torch.stack(tensor_x3, dim=0)
    
    return [tensor_x1, tensor_x2, tensor_x3], torch.LongTensor(tensor_y)


class CustomImageDataset(Dataset):
    def __init__(self, data, out_img=False):
        self.data = data
        self.transforms = transforms.Compose(
                [
                # transforms.ToTensor(),
                # transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
        self.out_img = out_img
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        tensor_x1 = torchvision.io.read_image(data[0]).float().div_(255)
        tensor_x2 = torchvision.io.read_image(data[1]).float().div_(255)
        tensor_x3 = torchvision.io.read_image(data[2]).float().div_(255)
        # tensor_x1 = torchvision.io.decode_jpeg(torchvision.io.read_file(data[0])).float().div_(255)
        # tensor_x2 = torchvision.io.decode_jpeg(torchvision.io.read_file(data[1])).float().div_(255)
        # tensor_x3 = torchvision.io.decode_jpeg(torchvision.io.read_file(data[2])).float().div_(255)
        tensor_x1_t = self.transforms(tensor_x1)
        tensor_x2_t = self.transforms(tensor_x2)
        tensor_x3_t = self.transforms(tensor_x3)
        if self.out_img: 
            return tensor_x1_t, tensor_x2_t, tensor_x3_t, torch.LongTensor([data[3]]), data[4], [tensor_x1, tensor_x2, tensor_x3]
        return tensor_x1_t, tensor_x2_t, tensor_x3_t, torch.LongTensor([data[3]]), data[4]
    