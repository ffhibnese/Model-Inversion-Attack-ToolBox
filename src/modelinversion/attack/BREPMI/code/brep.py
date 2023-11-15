import torch
from torch import nn
from dataclasses import dataclass
from ....utils import FolderManager
from .gen_z import init_z, decision
from tqdm import tqdm

@dataclass
class BrepArgs:
    
    sphere_points_count : int = 32
    init_sphere_radius : float = 2
    sphere_expansion_coeff : float = 1.3
    # step_size : float = 0
    # current_iter : 0
    point_clamp_min : float = -1.5
    point_clamp_max : float = 1.5
    max_iters_at_radius_before_terminate : int = 1000
    z_dim : int = 100
    # num_targets : int = 300
    batch_dim_for_initial_points : int = 256
    repulsion_only : bool = True
    # targeted_attack: bool = False
    
# Sample "#points_count" points around a sphere centered on "current_point" with radius =  "sphere_radius"
def gen_points_on_sphere(current_point, points_count, sphere_radius, device):
    
    # get random perturbations
    points_shape = (points_count,) + current_point.shape
    perturbation_direction = torch.randn(*points_shape).to(device)
    dims = tuple([i for i in range(1, len(points_shape))])
    
    # normalize them such that they are uniformly distributed on a sphere with the given radius
    perturbation_direction = (sphere_radius/ torch.sqrt(torch.sum(perturbation_direction ** 2, axis = dims, keepdims = True))) * perturbation_direction
    
    # add the perturbations to the current point
    sphere_points = current_point + perturbation_direction
    return sphere_points, perturbation_direction

def is_target_class(idens, target, model,score=False, criterion = None):
    if score:
        target_class_tensor = torch.tensor([target]).cuda()
        val_iden, score = decision(idens,model, score, target_class_tensor, criterion = criterion )
    else:
        val_iden = decision(idens,model)
        
    val_iden[val_iden != target] = 0
    val_iden[val_iden == target] = 1
    return val_iden
    
def brep_single_attack(args: BrepArgs, G, T, E, z, label, creterion, folder_manager: FolderManager, device) -> bool:
    current_iter = 0
    last_iter_when_radius_changed = 0
    
    G.eval()
    E.eval()
    T.eval()
    
    current_z = z.unsqueeze(0).to(device)
    # label_tensor = torch.LongTensor([label], device=device)
    
    current_sphere_radius = args.init_sphere_radius
    
    last_success_on_eval = False
    
    with torch.no_grad():
        while current_iter - last_iter_when_radius_changed < args.max_iters_at_radius_before_terminate:
            
            current_iter += 1
            
            new_radius = False
            
            step_size = min(current_sphere_radius / 3, 3)
            
            # sample points on the sphere
            new_points, perturbation_directions = gen_points_on_sphere(current_z[0], args.sphere_points_count, current_sphere_radius, device=device)
            
            # print(f">> {new_points.shape}")
            # exit()
            
            # get the predicted labels of the target model on the sphere points
            new_points_classification = is_target_class(G(new_points), label, T)
            
            if new_points_classification.sum() > 0.75 * args.sphere_points_count:
                new_radius = True
                last_iter_when_radius_changed = current_iter
                current_sphere_radius *= args.sphere_expansion_coeff
            
            # get the update direction, which is the mean of all points outside boundary if 'repulsion_only' is used. Otherwise it is the mean of all points * their classification (1,-1)
            if args.repulsion_only == True:
                new_points_classification = (new_points_classification - 1)/2
                
            grad_direction = torch.mean(new_points_classification.unsqueeze(1) * perturbation_directions, axis = 0) / current_sphere_radius
            
            # move the current point with stepsize towards grad_direction
            z_new = current_z + step_size * grad_direction
            z_new = z_new.clamp(min=args.point_clamp_min, max=args.point_clamp_max)
            
            current_img = G(z_new)
            
            if is_target_class(current_img, label,T)[0] == -1:
                # log_file.write("current point is outside target class boundary")
                continue
            
            current_z = z_new
            eval_res = decision(current_img, E)[0].item()
            # print(eval_res)
            # exit()
            correct_on_eval = eval_res==label
            
            if new_radius:
                # point_before_inc_radius = current_z.clone()
                last_success_on_eval = correct_on_eval
                continue
            
    folder_manager.save_result_image(current_img, label)
            
    return 1 if last_success_on_eval else 0
    

def brep_attack(args: BrepArgs, G, T, E, target_labels, folder_manager: FolderManager, device):
    init_z_ = init_z(args.batch_dim_for_initial_points, G, T, args.point_clamp_min, args.point_clamp_max, args.z_dim, target_labels, device=device)
    criterion = nn.CrossEntropyLoss().to(device)
    correct_on_eval = 0
    current_iter = 0
    
    for label in tqdm(init_z_.keys()):
        current_iter += 1
        z = init_z_[label]
        
        correct_on_eval += brep_single_attack(args, G, T, E, z, label, criterion, folder_manager, device=device)
        
    total_acc_on_eval = correct_on_eval / len(init_z_)
    
    
    
    return total_acc_on_eval
        
        
        