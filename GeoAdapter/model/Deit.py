# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, Mlp
from einops import rearrange
import numpy as np
from functools import partial
import torchvision

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.inv_act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_inv_fcb1 = nn.parameter.Parameter(torch.zeros(D_features))
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.D_inv_fcb2 = nn.parameter.Parameter(torch.zeros(D_hidden_features))
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    
    def forward_inv(self, x):
        xs = F.linear(x, self.D_fc2.weight.T, self.D_inv_fcb2)
        xs = self.inv_act(xs)
        xs = F.linear(xs, self.D_fc1.weight.T, self.D_inv_fcb1)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return xs


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            qk_scale=None
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_frames, num_heads, mlp_ratio=4., scale=0.5, num_tadapter=1, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_frames = num_frames
        self.num_tadapter = num_tadapter
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # self.MLP_Adapter = Adapter(dim, skip_connect=False)  # MLP-adapter, no skip connection
        # self.S_Adapter = Adapter(dim)  # with skip connection
        self.scale = scale
        self.T_Adapter = Adapter(dim)  # no skip connection
        # if num_tadapter == 2:
        #     self.T_Adapter_in = Adapter(dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, temporal_embed, mask):
        ## x in shape [BT, HW+2, D]
        bt, n, d = x.shape

        x = x + self.drop_path(self.attn(self.norm1(x)))

        ###
        xt = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        xt = xt + temporal_embed
        
        xt = self.T_Adapter(xt)
        
        xt = self.T_Adapter.forward_inv(self.attn(self.norm1(xt), mask=mask))
        xt = rearrange(xt, '(b n) t d -> (b t) n d', n=n)
        ###

        x = x + self.drop_path(xt)
        xn = self.norm2(x)
        x = x + self.mlp(xn)
        return x


class DeitAdapter(VisionTransformer):

    def __init__(
            self,
            num_frames,
            aerial,
            crop=False, 
            save=None,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_norm=False,
            init_values=None,
            class_token=True,
            no_embed_class=False,
            pre_norm=False,
            fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            weight_init='',
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            block_fn=Block,
    ):
        super(VisionTransformer, self).__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False
        self.num_frames = num_frames
        self.aerial = aerial

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                num_frames=num_frames,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Deit params
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        self.save = save
        self.crop = crop
        self.crop_rate = 0.64 # keep rate: 0.53 352, 0.64 320, 0.79 288

        if weight_init != 'skip':
            self.init_weights(weight_init)

        print(f"Model loaded with {self.aerial}")

    def init_weights(self, mode=''):
        super().init_weights(mode)
        ## initialize S_Adapter
        for n, m in self.blocks.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.blocks.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.blocks.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'temporal_embedding'}
    
    def forward(self, x, atten=None, indexes=None):
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)  # shape = [BT, HW, D]
        cls_tokens = self.cls_token.expand(B*T, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B*T, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.pos_embed.to(x.dtype)

        n = x.shape[1]
        mask = None
        if self.aerial:
            mask = ~torch.eye(T).to(torch.bool) # T, T
            mask = mask.repeat(n, 1, 1).repeat(B, 1, 1, 1) # B, N, T, T
            mask[:, 0:2, :, :] = False

            mask = rearrange(mask, 'b n t1 t2 -> (b n) t1 t2')
            mask = mask.unsqueeze(1).to(x.device)

        # x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        # x = x + self.temporal_embedding
        # x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
        
        for blk in self.blocks:
            x = blk(x, self.temporal_embedding, mask=mask)  # [BT, HW+2, D]

        x = self.norm(x)
        x_cls = self.head(x[:, 0])
        x_dist = self.head_dist(x[:, 1])
        x = (x_cls + x_dist)/2
        x = rearrange(x, '(b t) c -> b t c',b=B,t=T)
        
        # x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x.mean(axis=1)
    
    def freeze_layers(self, missing_keys):
        for n, p in self.named_parameters():
            if n not in missing_keys:
                p.requires_grad = False


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, crop=False, save=None, num_frames=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        self.save = save
        self.crop = crop
        self.crop_rate = 0.64 # keep rate: 0.53 352, 0.64 320, 0.79 288


    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward_features_crop(self, x, atten):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # add nonuniform-cropping

        B = x.shape[0]
        grid_size = (x.shape[-2] // self.patch_embed.patch_size[0], x.shape[-1] // self.patch_embed.patch_size[1])
        x = self.patch_embed(x)
        # sort based on attention
        atten_reshape = torch.nn.functional.interpolate(atten.detach(), grid_size, mode='bilinear')
        order = torch.argsort(atten_reshape[:,0,:,:].reshape([B,-1]),dim=1)
        # select patches
        select_list = []
        pos_list = []
        for k in range(B):
            select_list.append(x[k,order[[k],-int(self.crop_rate*order.shape[1]):]])
            pos_list.append(torch.cat([self.pos_embed[:,:2],self.pos_embed[:,2+order[k,-int(self.crop_rate*order.shape[1]):]]],dim=1))

        x = torch.cat(select_list,dim=0)
        pos_embed = torch.cat(pos_list,dim=0)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]


    def forward(self, x, atten=None, indexes=None):
        if self.save is not None:
            x, x_dist = self.forward_features_save(x, indexes)
        elif self.crop:
            if atten is None:
                atten = torch.zeros_like(x).cuda()
            x, x_dist = self.forward_features_crop(x, atten)
        else:
            x, x_dist = self.forward_features(x)

        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        # follow the evaluation of deit, simple average and no distillation during training, could remove the x_dist
        return (x + x_dist) / 2
    
    def freeze_layers(self, missing_keys):
        for n, p in self.named_parameters():
            if n not in missing_keys:
                p.requires_grad = False

@register_model
def deit_small_distilled_patch16_224(adapter=False, aerial=False, pretrained=True, img_size=(224,224), num_classes =1000, **kwargs):
    if adapter:
        model_callable = partial(DeitAdapter, aerial=aerial)
    else:
        model_callable = DistilledVisionTransformer
    model = model_callable(
        img_size=img_size, patch_size=16, embed_dim=384, num_classes=num_classes, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        # for key in checkpoint["model"]:
        #     print(key)

        # resize the positional embedding
        weight = checkpoint["model"]['pos_embed']
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
        new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
        matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
        resize = torchvision.transforms.Resize(new_size)
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
        checkpoint["model"]['pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)
        # change the prediction head if not 1000
        if num_classes != 1000:
            checkpoint["model"]['head.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            checkpoint["model"]['head_dist.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head_dist.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
        
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        assert len(msg.unexpected_keys) == 0, "unexpected keys: {}".format(msg.unexpected_keys)
        # print('no pretrain weights, init total random')
        # missing_keys = []
        # for n, p in model.named_parameters():
        #     if n not in checkpoint["model"]:
        #         missing_keys.append(n)
    return model, msg.missing_keys
    # return model, missing_keys


if __name__ == "__main__":
    model, missing_keys = deit_small_distilled_patch16_224(num_frames=8, adapter=True, aerial=True)

    inp = torch.randn((1, 8, 3, 224, 224))
    out = model(inp)
    print(out.shape)
    # print(len(missing_keys))
    # for n, p in model.named_parameters():
    #     if n not in missing_keys:
    #         p.requires_grad = False

    # cnt = 0
    # for p in model.parameters():
    #     if p.requires_grad:
    #         cnt += 1
    # print(cnt)