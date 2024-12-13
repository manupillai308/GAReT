from .TransGeo import TransGeo
import torch
from ptflops import get_model_complexity_info

def compute_complexity(model, args):
    size_sat = [256, 256]  # [512, 512]
    size_sat_default = [256, 256]  # [512, 512]
    size_grd = [216, 384]  # [224, 1232]

    if args.sat_res != 0:
        size_sat = [args.sat_res, args.sat_res]

    if args.fov != 0:
        size_grd[1] = int(args.fov /360. * size_grd[1])
# query flops: 257.11026632 + 11.136434768 + 0.9447512
    with torch.cuda.device(0):
        macs_1, params_1 = get_model_complexity_info(model.query_net, input_res=(8, 3, size_grd[0], size_grd[1]),#input_constructor=lambda x: {'x':torch.randn((38, 3, size_grd[0], size_grd[1])).cuda()},
        as_strings=False, print_per_layer_stat=False, verbose=False)
        # macs_2, params_2 = get_model_complexity_info(model.reference_net, (3, 1792 , 1792),
        #                                              as_strings=False,
        #                                              print_per_layer_stat=False, verbose=False)

        # print('flops:', (macs_1+macs_2)/1e9, 'params:', (params_1+params_2)/1e6)
        print('query flops:', (macs_1)/1e9)
        # print('ref flops:', (macs_2)/1e9)
        
    
if __name__ == "__main__":
    class Args:
        dataset='gama'
        dim=1000
        sat_res=0
        fov=0
        crop=False
        type='sym2'

    args = Args()
    import time
    with torch.no_grad():
        model = TransGeo(num_frames=8, args=args).cuda()
        print("Trainable params:", sum(
        p.numel() for p in model.parameters() if p.requires_grad))
        torch.cuda.reset_peak_memory_stats(device=0)
        s = time.time()
        model.query_net(torch.randn(1, 8, 3, 216, 384, dtype=torch.float32).cuda())
        print(f"Running time:", time.time() - s, "seconds")
        print(f"gpu used {torch.cuda.max_memory_allocated(device=0)/1024**3} GB memory")
        # compute_complexity(model, args)
# 0.239