import torch
torch.set_grad_enabled(False)
from torch import nn
import torchvision
from torchvision.transforms.functional import to_pil_image
import numpy as np
from types import SimpleNamespace
from model.TransGeo import TransGeo
import os
from functools import partial
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from dataset.GAMa import Patchify, input_transform_q, input_transform
import pandas as pd
import time
from tqdm import tqdm

class InferenceDL(torch.utils.data.Dataset):
    def __init__(self, grd_root, aerial_grd_root, aerial_root, city):
        self.grd_root = grd_root
        self.aerial_root = aerial_root
        if city == 'all':
            self.folder = list(set(os.listdir(aerial_grd_root)).intersection(set(os.listdir(grd_root))))
        else:
            df = pd.read_csv('/home/c3-0/sarucrcv/geo3/BDD100k_Big/metadata.csv')
            folder = df[(df['classified_into'] == city) & (df['train_or_val'] == 0)]['file_name'].tolist()
            if len(folder) == 0:
                folder = pd.read_csv(f'./{city}.csv', header=None)[0].tolist()
            self.folder = list(set(os.listdir(aerial_grd_root)).intersection(set(folder)))
        self.sat_size = [256, 256]
        self.grd_size = [216, 384]
        self.transform_query = input_transform_q(size=self.grd_size)
        self.transform_reference = input_transform(size=self.sat_size)
        
    def load_img(self, imgs, transform):
        out = [transform(Image.open(img)).unsqueeze(0) for img in imgs]
        return torch.cat(out, axis=0)
    
    def __len__(self):
        return len(self.folder)
    
    def __getitem__(self, i):
        folder = self.folder[i]
        folder_path = os.path.join(self.grd_root, folder)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        img_q = self.load_img(files[::5], self.transform_query)
        img_r = Image.open(os.path.join(self.aerial_root, folder+'.jpeg')).convert('RGB')
        img_r = self.transform_reference(img_r)
        return img_q, img_r, folder
    
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
# inv_t = UnNormalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])

def load_checkpoint(model, img_size, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')['state_dict']
    weight = checkpoint['reference_net.pos_embed']
    ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
    new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
    matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
    resize = torchvision.transforms.Resize(new_size)
    new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
    checkpoint['reference_net.pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)

    return checkpoint

# def show_img(tensor, nrow):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(to_pil_image(inv_t(torchvision.utils.make_grid(tensor, nrow=nrow))))
#     plt.axis('off')
#     plt.show()
    
# def to_image(tensor, nrow):
#     return to_pil_image(inv_t(torchvision.utils.make_grid(tensor, nrow=nrow)))
def accuracy(query_features, reference_features, query_labels, topk=[1,5,10]):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    ts = time.time()
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])
    sim_indexs = np.empty((N, M), dtype=np.int32)
    query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
    reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
    similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())

    for i in range(N):
        sim_indexs[i] = np.argsort(similarity[i, :])[::-1]
        ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)

        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.

    results = results/ query_features.shape[0] * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
    return results[:2], sim_indexs


if __name__ == "__main__":
    import sys
    mode = 'val'
    print(f'Running mode {mode}')
    city = sys.argv[1] if len(sys.argv) > 1 else 'all'
    print(f'Running inference on {city}')
    if os.path.exists(f'embeddings-geo-adapter-avg-{city}-{mode}.csv'):
        print(f"Found embeddings-geo-adapter-avg-{city}-{mode}.csv")
        feat_arr = pd.read_csv(f'embeddings-geo-adapter-avg-{city}-{mode}.csv').values
        print(feat_arr.shape)
        folders = feat_arr[:, 0]
        feat_arr = feat_arr[:, 1:].astype('float')
        feat_q, feat_r = feat_arr[:, :1000], feat_arr[:, 1000:]
        labels = np.arange(len(feat_q))
        _, sim_indexs = accuracy(feat_q, feat_r, query_labels=labels)
        with open(f"predictions-geo-adapter-avg-{city}-{mode}.txt", "w") as f:
            for i in range(sim_indexs.shape[0]):
                f.write(",".join([folders[i]] + list(folders[sim_indexs[i]])) + "\n")
    else:
        args = SimpleNamespace(asam=True, batch_size=6, city='all', cos=True, crop=False, cross=False, dataset='gama', dim=1000, dist_backend='nccl', dist_url='tcp://localhost:10003', epochs=100, evaluate=False, fov=0, gpu=None, lr=0.0001, mining=True, momentum=0.9, multiprocessing_distributed=True, op='sam', print_freq=10, rank=0, resume='', rho=2.5, sat_res=0, save_path='./result_GAMA_GeoAdapter_x', schedule=[120, 160], seed=None, share=False, start_epoch=0, weight_decay=0.03, workers=10, world_size=1, type='avg')
        # args.distributed = args.world_size > 1 or args.multiprocessing_distributed
        # args.gpu = torch.cuda.current_device()
        device = torch.device('cuda:0')
        model = TransGeo(num_frames=8, args=args)
        ckpt_path='/home/ma293852/Project/TransGeo2022-video/result_GAMA_GeoAdapter_AVG/model_best.pth.tar'
        checkpoint = load_checkpoint(model=model.reference_net, img_size=model.size_sat, ckpt_path=ckpt_path)
        model.load_state_dict(checkpoint, strict=True)
        model.to(device)
        
        dataset = InferenceDL(grd_root=f'/home/c3-0/sarucrcv/geo3/BDD100k_Big/Ground/{mode}/', aerial_grd_root=f'/home/ma293852/Project/dataset/GAMa/{mode}_gps/', aerial_root=f'/home/ma293852/Project/dataset/GAMa/{mode}_hr/', city=city)
        
        print(len(dataset))
        mapping = {folder:idx for folder,idx in zip(dataset.folder, range(len(dataset)))}
        feat_arr = np.empty((len(dataset), 2000))
        indexes = [None]*len(dataset)
        with torch.no_grad():
            for folder, idx in mapping.items():
                indexes[idx] = folder
            for i, (img_q, img_r, folder) in tqdm(enumerate(dataset), total=len(dataset)):
                feat_q, feat_r = model(img_q.unsqueeze(0).to(device), img_r.unsqueeze(0).to(device))
                feat_arr[i, :1000] = feat_q.cpu().numpy()
                feat_arr[i, 1000:] = feat_r.cpu().numpy()
        
        pd.DataFrame(feat_arr, index=indexes).to_csv(f'embeddings-geo-adapter-avg-{city}-{mode}.csv')