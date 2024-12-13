import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import random



def input_transform_q(size, fov=None):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])
    

def input_transform(size, fov=None):
    return transforms.Compose([
        transforms.CenterCrop(size=(256, 256)),
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

input_transform_fov = input_transform
input_transform_fov_q = input_transform_q

# Same loader from VIGOR, modified for pytorch
class BDD(torch.utils.data.Dataset):
    def __init__(self, mode = '', root = '/home/c3-0/sarucrcv/geo3/BDD100k_Big/', same_area=True, print_bool=False, polar = '', csv_root='/home/ma293852/Project/TransGeo2022', args=None):
        super(BDD, self).__init__()

        self.args = args
        self.root = root
        self.polar = polar
        self.csv_root = csv_root

        self.mode = mode
        self.sat_size = [256, 256]
        self.sat_size_default = [320, 320]
        self.grd_size = [216, 384]
        if args.sat_res != 0:
            self.sat_size = [args.sat_res, args.sat_res]
        if print_bool:
            print(self.sat_size, self.grd_size)

        self.sat_ori_size = [750, 750]
        self.grd_ori_size = [720, 1280]

        if args.fov != 0:
            self.transform_query = input_transform_fov_q(size=self.grd_size,fov=args.fov)
        else:
            self.transform_query = input_transform_q(size=self.grd_size)

        if len(polar) == 0:
            self.transform_reference = input_transform(size=self.sat_size)
        else:
            self.transform_reference = input_transform(size=self.sat_ori_size)
        self.to_tensor = transforms.ToTensor()
        self.train_csv = open(os.path.join(self.csv_root, "train_GAMA.csv")).readlines()
        self.test_csv = open(os.path.join(self.csv_root, "val_GAMA.csv")).readlines()


        self.train_csv = list(filter(lambda x: os.path.exists(os.path.join(self.root, "Ground", "train", x.strip())), self.train_csv))
        self.test_csv = list(filter(lambda x: os.path.exists(os.path.join(self.root, "Ground", "val", x.strip())), self.test_csv))
        

        self.train_sat_list = []
        self.train_sat_index_dict = {}
        self.delta_unit = [0.0003280724526376747, 0.00043301140280175833]
        idx = 0
        # load sat list
        for folder in self.train_csv:
            folder = folder.strip()
            imgs = sorted(os.listdir(os.path.join(self.root, "Aerial", "train", folder)))
            mid = 2
            while mid < len(imgs):
                img = imgs[mid]
                sat_name = '_'.join(img.split('_')[1:])
                self.train_sat_list.append(os.path.join(self.root, "Aerial", "train", folder, img))
                self.train_sat_index_dict[sat_name] = idx
                mid += 5
                idx += 1 
        if print_bool:
            print('InputData::__init__: load', len(self.train_csv), idx)
            
        self.train_sat_list = np.array(self.train_sat_list)
        self.train_sat_data_size = len(self.train_sat_list)
        if print_bool:
            print('Train sat loaded, data size:{}'.format(self.train_sat_data_size))

        self.test_sat_list = []
        self.test_sat_index_dict = {}
        idx = 0
        for folder in self.test_csv:
            folder = folder.strip()
            imgs = sorted(os.listdir(os.path.join(self.root, "Aerial", "val", folder)))
            mid = 0
            while mid < len(imgs):
                img = imgs[mid]
                sat_name = '_'.join(img.split('_')[1:])
                self.test_sat_list.append(os.path.join(self.root, "Aerial", "val", folder, img))
                self.test_sat_index_dict[sat_name] = idx
                mid += 1
                idx += 1 
        if print_bool:
            print('InputData::__init__: load', len(self.test_csv), idx)

        self.test_sat_list = np.array(self.test_sat_list)
        self.test_sat_data_size = len(self.test_sat_list)
        if print_bool:
            print('Test sat loaded, data size:{}'.format(self.test_sat_data_size))

        self.train_list = []
        self.train_label = []
        self.train_sat_cover_dict = {}
        self.train_delta = []
        idx = 0

        for folder in self.train_csv:
            folder = folder.strip()
            imgs = sorted(os.listdir(os.path.join(self.root, "Ground", "train", folder)))
            indexs = list(zip(range(0, len(imgs), 5), range(5, len(imgs)+5, 5)))
            mids = list(range(2, len(imgs), 5))
            for ((st,end), mid) in zip(indexs, mids):
                # grd_pair_imgs = [os.path.join(self.root, "Ground", "train", folder, img) for img in imgs[st:end]]
                grd_pair_imgs = [os.path.join(self.root, "Ground", "train", folder, img) for img in [imgs[mid]]]
                img = imgs[mid]
                sat_pair_name = '_'.join(img.split('_')[1:])
                label = []
                for i in [1]:
                    label.append(self.train_sat_index_dict[sat_pair_name])
                label = np.array(label).astype(int)
                delta = np.array([0]).astype(float)
                self.train_list.extend(grd_pair_imgs)
                ll = len(grd_pair_imgs)
                self.train_sat_cover_dict[sat_pair_name] = list(range(idx, idx+ll))
                self.train_label.extend([label]*ll)
                self.train_delta.extend([delta]*ll)
                idx += ll
        if print_bool:
            print('InputData::__init__: load ', len(self.train_csv), idx)

        self.train_data_size = len(self.train_list)
        self.train_label = np.array(self.train_label)
        self.train_delta = np.array(self.train_delta)
        if print_bool:
            print('Train grd loaded, data_size: {}'.format(self.train_data_size))

        self.test_list = []
        self.test_label = []
        self.test_sat_cover_dict = {}
        self.test_delta = []
        idx = 0

        for folder in self.test_csv:
            folder = folder.strip()
            imgs = sorted(os.listdir(os.path.join(self.root, "Ground", "val", folder)))
            indexs = list(zip(range(0, len(imgs), 1), range(1, len(imgs)+1, 1)))
            # mids = list(range(2, len(imgs), 5))
            mids = list(range(0, len(imgs), 1))
            for ((st,end), mid) in zip(indexs, mids):
                # grd_pair_imgs = [os.path.join(self.root, "Ground", "val", folder, img) for img in imgs[st:end]]
                img = imgs[mid]
                grd_pair_imgs = [os.path.join(self.root, "Ground", "val", folder, img)]
                sat_pair_name = '_'.join(img.split('_')[1:])
                label = []
                for i in [1]:
                    label.append(self.test_sat_index_dict[sat_pair_name])
                label = np.array(label).astype(int)
                delta = np.array([0]).astype(float)
                self.test_list.extend(grd_pair_imgs)
                ll = len(grd_pair_imgs)
                self.test_sat_cover_dict[sat_pair_name] = list(range(idx, idx+ll))
                self.test_label.extend([label]*ll)
                self.test_delta.extend([delta]*ll)
                idx += ll
        if print_bool:
            print('InputData::__init__: load ', len(self.train_csv), idx)

        self.test_data_size = len(self.test_list)
        self.test_label = np.array(self.test_label)
        self.test_delta = np.array(self.test_delta)
        if print_bool:
            print('Test grd loaded, data size: {}'.format(self.test_data_size))

        self.train_sat_cover_list = list(self.train_sat_cover_dict.keys())


    def check_overlap(self, id_list, idx):
        output = True
        sat_idx = self.train_label[idx]
        for id in id_list:
            sat_id = self.train_label[id]
            for i in sat_id:
                 if i in sat_idx:
                    output = False
                    return output
        return output

    def get_init_idx(self):
        # return random.randrange(self.train_data_size)  # sampling according to grd
        return random.choice(self.train_sat_cover_dict[random.choice(self.train_sat_cover_list)])

    def __getitem__(self, index, debug=False):
        if 'train' in self.mode:
            if 'scan' in self.mode:
                # replace random sample with deterministic sample if it is to scan all the samples
                ll = len(self.train_sat_cover_dict[self.train_sat_cover_list[index % len(self.train_sat_cover_list)]])
                assert ll <= 5
                idx = self.train_sat_cover_dict[self.train_sat_cover_list[index%len(self.train_sat_cover_list)]][(index//len(self.train_sat_cover_list))%ll]
            else:
                idx = random.choice(self.train_sat_cover_dict[self.train_sat_cover_list[index%len(self.train_sat_cover_list)]])
            img_query = Image.open(self.train_list[idx])
            img_reference = Image.open(self.train_sat_list[self.train_label[idx][0]]).convert('RGB')

            img_query = self.transform_query(img_query)
            img_reference = self.transform_reference(img_reference)
            if self.args.crop:
                atten_sat = Image.open(os.path.join(self.args.resume.replace(self.args.resume.split('/')[-1],''),'attention','train', str(idx)+'.png')).convert('RGB')
                return img_query, img_reference, torch.tensor(idx), torch.tensor(idx), torch.tensor(self.train_delta[idx, 0]), self.to_tensor(atten_sat)
            return img_query, img_reference, torch.tensor(idx), torch.tensor(idx), torch.tensor(self.train_delta[idx, 0]), 0
        elif 'scan_val' in self.mode:
            img_reference = Image.open(self.test_sat_list[index]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            img_query = random.choice(self.test_list)
            img_query = Image.open(img_query)
            img_query = self.transform_query(img_query)
            return img_query, img_reference, torch.tensor(index), torch.tensor(index), 0, 0
        elif 'test_reference' in self.mode:
            img_reference = Image.open(self.test_sat_list[index]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            if self.args.crop:
                atten_sat = Image.open(os.path.join(self.args.resume.replace(self.args.resume.split('/')[-1],''),'attention','val', str(index)+'.png')).convert('RGB')
                return img_reference, torch.tensor(index), self.to_tensor(atten_sat)
            return img_reference, torch.tensor(index), 0
        elif 'test_query' in self.mode:
            img_query = Image.open(self.test_list[index])
            img_query = self.transform_query(img_query)
            return img_query, torch.tensor(index), torch.tensor(self.test_label[index][0])
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        if 'train' in self.mode:
            return len(self.train_sat_cover_list) * 5  # one aerial image has 5 positive queries
        elif 'scan_val' in self.mode:
            return len(self.test_sat_list)
        elif 'test_reference' in self.mode:
            return len(self.test_sat_list)
        elif 'test_query' in self.mode:
            return len(self.test_list)
        else:
            print('not implemented!')
            raise Exception


# compute the distance between two locations [Lat_A,Lng_A], [Lat_B,Lng_B]
def gps2distance(Lat_A,Lng_A,Lat_B,Lng_B):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    lat_A = Lat_A * np.pi/180.
    lat_B = Lat_B * np.pi/180.
    lng_A = Lng_A * np.pi/180.
    lng_B = Lng_B * np.pi/180.
    R = 6371004.
    C = np.sin(lat_A)*np.sin(lat_B) + np.cos(lat_A)*np.cos(lat_B)*np.cos(lng_A-lng_B)
    distance = R*np.arccos(C)
    return distance


# compute the distance between two locations [Lat_A,Lng_A], [Lat_B,Lng_B]
def gps2distance_matrix(Lat_A,Lng_A,Lat_B,Lng_B):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    lat_A = Lat_A * np.pi/180.
    lat_B = Lat_B * np.pi/180.
    lng_A = Lng_A * np.pi/180.
    lng_B = Lng_B * np.pi/180.
    R = 6371004.
    C1 = np.matmul(np.sin(np.expand_dims(lat_A,axis=1)), np.sin(np.expand_dims(lat_B,axis=0)))
    C2 = np.matmul(np.cos(np.expand_dims(lat_A,axis=1)),np.cos(np.expand_dims(lat_B,axis=0)))
    C2 = C2 * np.cos(np.tile(np.expand_dims(lng_A,axis=1),[1,lng_B.shape[0]])-np.tile(lng_B,[np.expand_dims(lng_A,axis=0).shape[0],1]))
    C = C1 + C2
    distance = R*np.arccos(C)
    return distance


# compute the delta unit for each reference location [Lat, Lng], 320 is half of the image width
# 0.114 is resolution in meter
# reverse equation from gps2distance: https://en.wikipedia.org/wiki/Great-circle_distance
def Lat_Lng(Lat_A, Lng_A, distance=[320*0.114, 320*0.114]):
    if distance[0] == 0 and distance[1] == 0:
        return np.zeros(2)

    lat_A = Lat_A * np.pi/180.
    lng_A = Lng_A * np.pi/180.
    R = 6371004.
    C_lat = np.cos(distance[0]/R)
    C_lng = np.cos(distance[1]/R)
    delta_lat = np.arccos(C_lat)
    delta_lng = np.arccos((C_lng-np.sin(lat_A)*np.sin(lat_A))/np.cos(lat_A)/np.cos(lat_A))
    return np.array([delta_lat * 180. / np.pi, delta_lng * 180. / np.pi])



if __name__ == "__main__":
    class Args:
        fov=0
        city="SanFrancisco"
        sat_res = 0
        crop = False
    args = Args()
    dataset = BDD(mode='train', args=args)
    dataloader = torch.utils.data.DataLoader(dataset)
    print(len(dataset))
    # for output in dataloader:
    #     print(output)
    #     break
    # print(dataset.train_sat_cover_dict)