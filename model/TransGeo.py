import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from .Deit import deit_small_distilled_patch16_224
import torchvision

class TransGeo(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self,  num_frames, args, base_encoder=None):
        """
        dim: feature dimension (default: 512)
        """
        super(TransGeo, self).__init__()
        self.dim = args.dim

        # create the encoders
        # num_classes is the output fc dimension

        if args.dataset == 'vigor':
            self.size_sat = [320, 320]
            self.size_sat_default = [320, 320]
            self.size_grd = [320, 640]
        elif args.dataset == 'cvusa':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [112, 616]
        elif args.dataset == 'cvact':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [112, 616]
        elif args.dataset in ['bdd', 'gama']:
            self.size_sat = [256, 256]
            self.size_sat_default = [320, 320]
            self.size_grd = [216, 384]

        if args.sat_res != 0:
            self.size_sat = [args.sat_res, args.sat_res]
        if args.fov != 0:
            self.size_grd[1] = int(args.fov / 360. * self.size_grd[1])

        self.ratio = self.size_sat[0]/self.size_sat_default[0]
        base_model = deit_small_distilled_patch16_224
        self.model_type = args.type

        if args.type == 'asym': # aerial has CLS, street has original
            self.query_net, missing_query = base_model(adapter=True, aerial=False, crop=False, img_size=self.size_grd, num_classes=args.dim, num_frames=num_frames)
            self.reference_net, missing_ref = base_model(adapter=True, aerial=True, crop=args.crop, img_size=self.size_sat, num_classes=args.dim, num_frames=49)
        elif args.type == 'sym1': # original
            self.query_net, missing_query = base_model(adapter=True, aerial=False, crop=False, img_size=self.size_grd, num_classes=args.dim, num_frames=num_frames)
            self.reference_net, missing_ref = base_model(adapter=True, aerial=False, crop=args.crop, img_size=self.size_sat, num_classes=args.dim, num_frames=49)
        elif args.type == 'sym2': # both has CLS
            self.query_net, missing_query = base_model(adapter=True, aerial=True, crop=False, img_size=self.size_grd, num_classes=args.dim, num_frames=num_frames)
            self.reference_net, missing_ref = base_model(adapter=True, aerial=True, crop=args.crop, img_size=self.size_sat, num_classes=args.dim, num_frames=49)
        elif args.type == 'avg':
            self.query_net, missing_query = base_model(adapter=False, aerial=False, crop=False, img_size=self.size_grd, num_classes=args.dim, num_frames=num_frames)
            self.reference_net, missing_ref = base_model(adapter=False, aerial=False, crop=args.crop, img_size=self.size_sat, num_classes=args.dim, num_frames=49)
        else:
            raise ValueError(f'invalid model type {args.type}')

        # if args.type != 'avg':
        #     self.query_net.freeze_layers(missing_query)
        #     self.reference_net.freeze_layers(missing_ref)
        self.polar = None

    def forward(self, im_q, im_k, delta=None, atten=None, indexes=None):
        if self.model_type == 'avg':
            B1, T1, C1, H1, W1 = im_q.shape
            B2, T2, C2, H2, W2 = im_k.shape
            query_feat = self.query_net(im_q.reshape(B1*T1, C1, H1, W1))
            reference_feat = self.reference_net(x=im_k.reshape(B2*T2, C2, H2, W2))
            return query_feat.reshape(B1, T1, self.dim).mean(dim=1), reference_feat.reshape(B2, T2, self.dim).mean(dim=1)
        if atten is not None:
            return self.query_net(im_q), self.reference_net(x=im_k, atten=atten)
        else:
            return self.query_net(im_q), self.reference_net(x=im_k, indexes=indexes)


if __name__ == "__main__":
    class Args:
        dataset='gama'
        dim=1000
        sat_res=0
        fov=0
        crop=False
        type='avg'

    args = Args()
    model = TransGeo(num_frames=8, args=args)
    # dp_model = torch.nn.DataParallel(model)
    # checkpoint = torch.load('/home/ma293852/Project/TransGeo2022/result_bdd_crop5_day/model_best.pth.tar', map_location=torch.device('cpu'))['state_dict']
    # weight = checkpoint['module.reference_net.pos_embed']
    # model = dp_model.module.reference_net
    # img_size = dp_model.module.size_sat
    # ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
    # new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
    # matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
    # resize = torchvision.transforms.Resize(new_size)
    # new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
    # checkpoint['module.reference_net.pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)

    # weight = checkpoint['module.query_net.pos_embed']
    # model = dp_model.module.query_net
    # img_size = dp_model.module.size_grd
    # ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
    # print(weight.shape)
    # new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
    # matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
    # resize = torchvision.transforms.Resize(new_size)
    # new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
    # checkpoint['module.query_net.pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)

    # msg = dp_model.load_state_dict(checkpoint, strict=False)
    # assert len(msg.unexpected_keys) == 0, "unexpected keys: {}".format(msg.unexpected_keys)
    # cnt = 0
    # for p in dp_model.parameters():
    #     if p.requires_grad:
    #         cnt += 1
    # print(cnt)

    # print(len(msg.missing_keys))

    print(model.model_type)
    g, a = model(torch.randn(2, 8, 3, 216, 384), torch.randn(2, 49, 3, 256, 256))
    print(g.shape, a.shape)
    