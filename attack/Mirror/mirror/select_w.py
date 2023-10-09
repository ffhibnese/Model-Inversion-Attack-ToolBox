import torch
import glob
import collections
import os
from attack.Mirror.utils.img_utils import *
from torch.nn import functional as F
from tqdm import tqdm

def find_closest_latent(target_model, image_resolution, targets_list, k, arch_name, pre_sample_dir, bs = 10):

    pre_device = next(target_model.parameters()).device
    
    device = 'cuda'# next(target_model.parameters()).device
    target_model = target_model.to(device)
    
    target_model.eval()
    
    target_ranked_confidence_dict = collections.defaultdict(list)
    ws = []
    
    img_dir = os.path.join(pre_sample_dir, 'img')
    w_dir = os.path.join(pre_sample_dir, 'w')
    
    all_ws_gen_files = sorted(glob.glob(os.path.join(w_dir, 'sample_*.pt')))
    all_img_gen_files = sorted(glob.glob(os.path.join(img_dir, 'sample_*.pt')))
    
    
    pt_num = 0
    
    with torch.no_grad():
        
        outputs = []
        
        for idx in tqdm(range(len(all_ws_gen_files))):
            
            # ws_file = all_ws_gen_files[idx]
            img_file = all_img_gen_files[idx]
            
            fake = torch.load(img_file).to(device)
            pt_num = len(fake)
            # w = torch.load(ws_file)
            fake = crop_img(fake, arch_name)
            fake = normalize(resize_img(fake*255., image_resolution), arch_name)
            
            # print(f'>>>>>>>>> target device {next(target_model.parameters()).device}\t fake device {fake.device}')
            
            
            for i in range(0, len(fake), bs):
                torch.cuda.empty_cache()
                output = F.softmax(target_model(fake[i:min(len(fake), i+bs)]).result, dim=-1)[:, targets_list]
                outputs.append(output.cpu())
                
        outputs = torch.cat(outputs, dim=0)
        
        for i, t in enumerate(targets_list):
            t_out = outputs[:, i]
            target_ranked_confidence_dict[t].append(t_out)
        # ws.append(w)
            
    target_ranked_confidence_dict = {t: torch.cat(v, dim=0) for t, v in target_ranked_confidence_dict.items()}
    # ws = torch.cat(ws, dim=0)
    
    # select top k
    
    res_w = collections.defaultdict(list)
    
    target_topk_conf_dict = {}
    target_topk_idx_dict = {}
    for t, v in target_ranked_confidence_dict.items():
        print(f'v: {v.shape}, k: {k}')
        topk_conf, topk_idx = torch.topk(v, k, dim=0)
        # print(f'{t}: {topk_conf}\t{topk_idx}')
        target_topk_idx_dict[t] = topk_idx.tolist()
        target_topk_conf_dict[t] = topk_conf
        
        for top_id in topk_idx:
            res_w[t].append(torch.load(all_ws_gen_files[top_id // pt_num])[top_id % pt_num])
            
        res_w[t] = torch.stack(res_w[t])
        
    
    # for t in targets_list:
    #     res[t] = (ws[target_topk_idx_dict[t]])
        
    target_model.to(pre_device)
    return res_w, target_topk_conf_dict
                
# a = collections.defaultdict(list)
# print(a[1])
# print(a)