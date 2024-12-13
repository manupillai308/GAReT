# %%
import os
import numpy as np
from models import *
import utm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical
import torch
from geopy.distance import distance
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from einops import rearrange
import datetime
import time
import gc
from builtins import print
from functools import partial

def get_lat_lon(f):
    f = f.strip()
    _, lat, lon = f.split('_')
    lon = lon.split('.jpg')[0]
    return float(lat), float(lon)

def recall_at_1(lats_pred, lons_pred, lats_gt, lons_gt, thresh=0.05):
    gt = list(zip(lats_gt, lons_gt))
    pred = list(zip(lats_pred, lons_pred))
    dist = np.empty(len(gt))
    for i in range(len(gt)):
        dist[i] = distance(gt[i], pred[i]).miles
    
    matches = (dist <= thresh)
    return matches

@torch.no_grad()
def validate(model, dataloader, thresh):
    model.eval()
    print(f'Validating with threshold {thresh}')

    matches_pred = []
    matches_nearest = []
    losses = []

    for step, data in (pbar := tqdm(enumerate(dataloader), total=len(dataloader))):
        
        x, labels, lats, lons, lats_gt, lons_gt = data
        print(x.shape)
        exit()
        x = x.to(device)
        
        # compute tours for model
        tours, ProbOfChoices = model(x, labels=None, deterministic=True) # size(tour_train)=(bsz, nb_nodes)
        indexes = torch.nonzero(labels.max(dim=2)[0]==1)
        # backprop
        loss = torch.nn.functional.nll_loss(input=ProbOfChoices[indexes[:, 0], indexes[:, 1]], target=labels.to(device).argmax(dim=2)[indexes[:, 0], indexes[:, 1]])
        losses.append(loss.item()) 
        matches_nearest.extend(recall_at_1(lats[..., -1].reshape(-1), lons[..., -1].reshape(-1), lats_gt.reshape(-1), lons_gt.reshape(-1), thresh=thresh))
        lats_pred = []
        lons_pred = []
        for b in range(lats.shape[0]):
            lats_pred.extend(lats[b][torch.arange(lats.shape[1]), tours.cpu()[b]])
            lons_pred.extend(lons[b][torch.arange(lons.shape[1]), tours.cpu()[b]])
        matches_pred.extend(recall_at_1(lats_pred, lons_pred, lats_gt.reshape(-1), lons_gt.reshape(-1), thresh=thresh))

    matches_pred = np.array(matches_pred)
    matches_nearest = np.array(matches_nearest)
    losses = np.array(losses)
    
    print(f'Validation: Nearest R@{thresh} = {matches_nearest.mean()*100:.3f}, Prediction R@{thresh} = {matches_pred.mean()*100:.3f}, Loss = {losses.mean():.3f}')
    return matches_pred.mean(), losses.mean()
    

    

# %%
class GeoGraphs(Dataset):
    def __init__(self, root, mode, topk=50, transform=None):
        super().__init__()
        self.root = root
        self.mode = mode
        self.topk = topk
        self.files_ = sorted(os.listdir(self.root))
        print(f'Total Size: {len(self.files_)}')
        l = len(self.files_)
        self.files_ = list(
        filter(lambda x: not os.path.exists(os.path.join(self.root, x, 'incorrect.f')), 
                self.files_))
        print(f'Final Size for mode {self.mode}: {len(self.files_)}')
        
        # with open(f'files_{self.mode}.txt', 'w') as f:
        #     f.write('\n'.join(self.files_))
        
        self.files_ = np.array(self.files_).astype(np.string_)
    
    def __len__(self):
        return len(self.files_)

    def __getitem__(self, idx):
        filename = self.files_[idx].decode()
        sim = np.loadtxt(os.path.join(self.root, filename, 'sim.txt'), dtype=np.float32)
        topk = np.argsort(sim, axis=1)[..., -self.topk:]
        
        y = np.loadtxt(os.path.join(self.root, filename, 'labels.txt'), dtype=np.float32)
        lats_orig = np.concatenate(np.loadtxt(os.path.join(self.root, filename, 'lats.txt')))
        lons_orig = np.concatenate(np.loadtxt(os.path.join(self.root, filename, 'lons.txt')))
        lats = np.repeat(
               lats_orig.reshape(-1,1),
            sim.shape[0], axis=1).T
        lons = np.repeat(
            lons_orig.reshape(-1,1),
            sim.shape[0], axis=1).T
        
        sim = sim[np.arange(sim.shape[0])[:, None], topk]
        lats = lats[np.arange(sim.shape[0])[:, None], topk].reshape(-1, 1)
        lons = lons[np.arange(sim.shape[0])[:, None], topk].reshape(-1, 1)
        # if self.mode == 'train':
        labels = topk == (np.arange(len(y))[y==1, None])
        
        easting, northing, _, _ = utm.from_latlon(lats, lons)
        sim, easting, northing = sim, easting.reshape(*sim.shape), northing.reshape(*sim.shape)
        x = StandardScaler().fit_transform(np.concatenate(np.dstack([sim, northing, easting])))

        lats_gt = np.loadtxt(os.path.join(self.root, filename, 'lats_gt.txt'), dtype=np.float32)
        lons_gt = np.loadtxt(os.path.join(self.root, filename, 'lons_gt.txt'), dtype=np.float32)
        
        
        return (torch.from_numpy(x.reshape(*sim.shape, 3).astype(np.float32)), 
                # torch.from_numpy(labels.astype(np.int32)) if self.mode == 'train' else [0], 
                torch.from_numpy(labels.astype(np.int32)), 
                torch.from_numpy(lats.reshape(*sim.shape).astype(np.float32)),
                torch.from_numpy(lons.reshape(*sim.shape).astype(np.float32)),
                torch.from_numpy(lats_gt.astype(np.float32)), torch.from_numpy(lons_gt.astype(np.float32))
               )

# %%
# x, labels, lats, lons, lats_gt, lons_gt = dataset[0]

# %%
# root='../../notebooks/evaluation/TransGeo2022-video-10/'
# dataset = GeoGraphs(root=root, mode='all', topk=10)

# # %%
# matches = []
# for i in tqdm(range(len(dataset))):
#     x, labels, lats, lons, lats_gt, lons_gt = dataset[i]
#     matches.extend(recall_at_1(lats[..., -1], lons[..., -1], lats_gt, lons_gt, thresh=0.01))
# np.mean(matches)



# %%
###################
# Hyper-parameters
###################
if __name__ == "__main__":
    class DotDict(dict):
        def __init__(self, **kwds):
            self.update(kwds)
            self.__dict__ = self
            
    args = DotDict()
    # args.nb_nodes = 20 # TSP20
    # args.nb_nodes = 50 # TSP50
    args.nb_nodes = 10
    #args.nb_nodes = 100 # TSP100
    args.bsz = 64 # TSP20 TSP50
    args.dim_emb = 128
    args.dim_ff = 512
    args.dim_input_nodes = 3
    args.nb_layers_encoder = 6
    args.nb_layers_decoder = 2
    args.nb_heads = 8
    args.nb_epochs = 10000
    args.nb_batch_per_epoch = 2500
    args.nb_batch_eval = 20
    args.gpu_id = '0'
    args.lr = 1e-4
    args.tol = 1e-3
    args.batchnorm = True  # if batchnorm=True  than batch norm is used
    #args.batchnorm = False # if batchnorm=False than layer norm is used
    args.max_len_PE = 1000
    args.thresh = 0.05
    # args.topk = 20
    args.topk = 10

    # print(args)

    # %%
    ###################
    # Instantiate a training network and a baseline network
    ###################
    # torch.autograd.set_detect_anomaly(True)

    model = TSP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, 
                args.nb_layers_encoder, args.nb_layers_decoder, args.nb_heads, args.max_len_PE,
                batchnorm=args.batchnorm)

    print('Loading checkpoint!')
    model.load_state_dict(torch.load('./checkpoint/24-02-15--16-48-28/checkpoint_15024-02-15--16-48-28-n10-gpu0.pkl')["model"])

    optimizer = torch.optim.Adam(model.parameters() , lr = args.lr ) 
    device = torch.device('cuda:0')
    model = model.to(device)


    # Logs
    os.makedirs('logs', exist_ok=True)
    time_stamp=datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
    file_name = 'logs'+'/'+time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + ".txt"
    file = open(file_name,"w",1) 
    file.write(time_stamp+'\n\n')
    for arg in vars(args):
        file.write(arg)
        hyper_param_val="={}".format(getattr(args, arg))
        file.write(hyper_param_val)
        file.write('\n')
    file.write('\n\n') 
    plot_performance_train = []
    plot_performance_baseline = []
    all_strings = []
    epoch_ckpt = 0
    tot_time_ckpt = 0

    # print = partial(print, file=file)


    # # Uncomment these lines to re-start training with saved checkpoint
    # checkpoint_file = "checkpoint/checkpoint_21-03-01--17-25-00-n50-gpu0.pkl"
    # checkpoint = torch.load(checkpoint_file, map_location=device)
    # epoch_ckpt = checkpoint['epoch'] + 1
    # tot_time_ckpt = checkpoint['tot_time']
    # plot_performance_train = checkpoint['plot_performance_train']
    # plot_performance_baseline = checkpoint['plot_performance_baseline']
    # model_baseline.load_state_dict(checkpoint['model_baseline'])
    # model_train.load_state_dict(checkpoint['model_train'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # print('Re-start training with saved checkpoint file={:s}\n  Checkpoint at epoch= {:d} and time={:.3f}min\n'.format(checkpoint_file,epoch_ckpt-1,tot_time_ckpt/60))
    # del checkpoint
    # # Uncomment these lines to re-start training with saved checkpoint

    # %%
    ###################
    # Main training loop 
    ###################
    start_training_time = time.time()
    train_dataset = GeoGraphs(root='/media/ma293852/New Volume/GeoAdapter/TransGeo2022-video-geoadapter-3-10-train', mode='train', topk=args.topk)
    test_dataset = GeoGraphs(root='/media/ma293852/New Volume/GeoAdapter/TransGeo2022-video-geoadapter-3-10-val', mode='test', topk=args.topk)

    train_dataloader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, num_workers=3)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bsz, num_workers=3)

    # print('Random validate epoch:', -1)
    recall = validate(model, test_dataloader, thresh=args.thresh)
    exit()
    for epoch in range(0,args.nb_epochs):
        
        # re-start training with saved checkpoint
        epoch += epoch_ckpt

        ###################
        # Train model for one epoch
        ###################
        start = time.time()
        model.train()
        losses = []

        for step, data in (pbar := tqdm(enumerate(train_dataloader), total=len(train_dataloader))):
            
            x, labels, lats, lons, lats_gt, lons_gt = data
            x = x.to(device)
            labels = labels.to(device)
            
            # compute tours for model
            tour_train, ProbOfChoices = model(x, labels=labels, deterministic=True) # size(tour_train)=(bsz, nb_nodes), size(sumLogProbOfActions)=(bsz)
            indexes = torch.nonzero(labels.max(dim=2)[0]==1)
            # backprop
            loss = torch.nn.functional.nll_loss(input=ProbOfChoices[indexes[:, 0], indexes[:, 1]], target=labels.argmax(dim=2)[indexes[:, 0], indexes[:, 1]])
            losses.append(loss.item())
            pbar.set_description(f'Epoch: {epoch}, Loss: {sum(losses)/len(losses):.4f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        time_one_epoch = time.time()-start
        time_tot = time.time()-start_training_time + tot_time_ckpt

        

        if epoch % 5 == 0:
            print('Validating epoch:', epoch)
            recall, val_loss = validate(model, test_dataloader, thresh=args.thresh)
            if epoch % 50 == 0:
                print('Training scores:', epoch)
                validate(model, train_dataloader, thresh=args.thresh)
            # Saving checkpoint
            checkpoint_dir = os.path.join("checkpoint", time_stamp)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save({
                'recall': recall,
                'val_loss': val_loss,
                'epoch': epoch,
                'time': time_one_epoch,
                'tot_time': time_tot,
                'loss': sum(losses)/len(losses),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, '{}.pkl'.format(checkpoint_dir + f"/checkpoint_{epoch}" + time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id)))

        
    file.close()
    # %%


    # %%


    # %%
    # for folder in os.listdir(root):
    #     with open(os.path.join(root, folder, 'refs.txt')) as f:
    #         data = [line.split(',') for line in f.readlines()]
    #         gps = np.empty((len(data), len(data[0]), 2))
            
    #         for i in range(gps.shape[0]):
    #             for j in range(gps.shape[1]):
    #                 gps[i, j] = get_lat_lon(data[i][j])
            
    #         np.savetxt(os.path.join(root, folder, 'lats.txt'), gps[..., 0])
    #         np.savetxt(os.path.join(root, folder, 'lons.txt'), gps[..., 1])        


