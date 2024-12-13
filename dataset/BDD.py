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
        # transforms.CenterCrop(size=(256, 256)),
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

input_transform_fov = input_transform
input_transform_fov_q = input_transform_q

# Same loader from VIGOR, modified for pytorch
class BDD(torch.utils.data.Dataset):
    def __init__(self, mode = '', root = '/home/c3-0/sarucrcv/geo3/BDD100k_Big/Ground', same_area=True, print_bool=False, polar = '', gama_root='/home/ma293852/Project/dataset/GAMa', args=None):
        super(BDD, self).__init__()

        self.args = args
        self.root = root
        self.polar = polar
        self.gama_root = gama_root

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
        self.train_hr = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(self.gama_root, "train_hr"))]
        self.test_hr = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(self.gama_root, "val_hr"))]

        train_l = len(self.train_hr)
        test_l = len(self.test_hr)
        self.train_hr = list(filter(lambda x: os.path.exists(os.path.join(self.root, "train", x)), self.train_hr))
        self.test_hr = list(filter(lambda x: os.path.exists(os.path.join(self.root, "val", x)), self.test_hr))

        self.train_sat_list = []
        self.train_sat_index_dict = {}
        self.delta_unit = [0.0003280724526376747, 0.00043301140280175833]
        idx = 0
        # load sat list
        for folder in self.train_hr:
            filename = folder + ".jpeg"
            img = os.path.join(self.gama_root, "train_hr", filename)
            self.train_sat_list.append(img)
            if self.train_sat_index_dict.get(folder):
                print(folder)
            self.train_sat_index_dict[folder] = idx
            idx += 1
        if print_bool:
            print('InputData::__init__: load', idx)
            
        self.train_sat_list = np.array(self.train_sat_list)
        self.train_sat_data_size = len(self.train_sat_list)
        if print_bool:
            print('Train sat loaded, data size: {}, skipped: {}'.format(self.train_sat_data_size, train_l-self.train_sat_data_size))

        self.test_sat_list = []
        self.test_sat_index_dict = {}
        idx = 0
        for folder in self.test_hr:
            filename = folder + ".jpeg"
            img = os.path.join(self.gama_root, "val_hr", filename)
            self.test_sat_list.append(img)
            self.test_sat_index_dict[folder] = idx
            idx += 1
        if print_bool:
            print('InputData::__init__: load', idx)

        self.test_sat_list = np.array(self.test_sat_list)
        self.test_sat_data_size = len(self.test_sat_list)
        if print_bool:
            print('Test sat loaded, data size: {}, skipped: {}'.format(self.test_sat_data_size, test_l-self.test_sat_data_size))

        self.train_list = []
        self.train_label = []
        self.train_sat_cover_dict = {}
        self.train_delta = []
        idx = 0

        for folder in self.train_hr:
            imgs = sorted(os.listdir(os.path.join(self.root, "train", folder)))
            grd_pair_imgs = [os.path.join(self.root, "train", folder, img) for img in imgs[::5]]
            self.train_list.append(grd_pair_imgs)
            self.train_sat_cover_dict[folder] = [idx]
            label = []
            for i in [1]:
                label.append(self.train_sat_index_dict[folder])
            label = np.array(label).astype(int)
            delta = np.array([0]).astype(float)
            self.train_label.append(label)
            self.train_delta.append(delta)
            idx += 1
        # if print_bool:
        #     print('InputData::__init__: load ', self.train_csv, idx)

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

        for folder in self.test_hr:
            imgs = sorted(os.listdir(os.path.join(self.root, "val", folder)))
            grd_pair_imgs = [os.path.join(self.root, "val", folder, img) for img in imgs[::5]]
            self.test_list.append(grd_pair_imgs)
            self.test_sat_cover_dict[folder] = [idx]
            label = []
            for i in [1]:
                label.append(self.test_sat_index_dict[folder])
            label = np.array(label).astype(int)
            delta = np.array([0]).astype(float)
            self.test_label.append(label)
            self.test_delta.append(delta)
            idx += 1

        # if print_bool:
        #     print('InputData::__init__: load ', self.train_csv, idx)

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
    
    def load_img(self, imgs, transform):
        out = [transform(Image.open(img)).unsqueeze(0) for img in imgs]
        return torch.cat(out, axis=0)

    # def get_init_idx(self):
        # return random.randrange(self.train_data_size)  # sampling according to grd
        # return random.choice(self.train_sat_cover_dict[random.choice(self.train_sat_cover_list)])

    def __getitem__(self, index, debug=False):
        if 'train' in self.mode:
            if 'scan' in self.mode:
                pass    
            idx = self.train_sat_cover_dict[self.train_sat_cover_list[index%len(self.train_sat_cover_list)]][0]
            img_query = self.load_img(self.train_list[idx], self.transform_query)

            img_reference = Image.open(self.train_sat_list[self.train_label[idx][0]]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            if self.args.crop:
                atten_sat = Image.open(os.path.join(self.args.resume.replace(self.args.resume.split('/')[-1],''),'attention','train', str(idx)+'.png')).convert('RGB')
                return img_query, img_reference, torch.tensor(idx), torch.tensor(idx), torch.tensor(self.train_delta[idx, 0]), self.to_tensor(atten_sat)
            return img_query, img_reference, torch.tensor(idx), torch.tensor(idx), torch.tensor(self.train_delta[idx, 0]), 0
        elif 'scan_val' in self.mode:
            img_reference = Image.open(self.test_sat_list[index]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            img_query = random.choice(self.test_list)
            img_query = self.load_img(img_query, self.transform_query)
            return img_query, img_reference, torch.tensor(index), torch.tensor(index), 0, 0
        elif 'test_reference' in self.mode:
            img_reference = Image.open(self.test_sat_list[index]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            if self.args.crop:
                atten_sat = Image.open(os.path.join(self.args.resume.replace(self.args.resume.split('/')[-1],''),'attention','val', str(index)+'.png')).convert('RGB')
                return img_reference, torch.tensor(index), self.to_tensor(atten_sat)
            return img_reference, torch.tensor(index), 0
        elif 'test_query' in self.mode:
            img_query = self.load_img(self.test_list[index], self.transform_query)
            return img_query, torch.tensor(index), torch.tensor(self.test_label[index][0])
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        if 'train' in self.mode:
            return len(self.train_sat_cover_list)
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
        fov = 0
        sat_res = 0
        crop = False
    args = Args()
    dataset = BDD(mode='train', args=args, print_bool=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)
    print(len(dataset))
    for output in dataloader:
        print(output[3].shape, output[3])
        break
    # print(dataset.train_sat_cover_dict)