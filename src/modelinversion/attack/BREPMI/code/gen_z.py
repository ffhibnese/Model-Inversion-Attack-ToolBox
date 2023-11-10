import torch

def decision(imgs, model, score=False, target=None, criterion = None):

    with torch.no_grad():
        T_out = model(imgs).result
        val_iden = torch.argmax(T_out, dim=1).view(-1)

    if score:
        return val_iden,criterion(T_out, target)
    else:
        return val_iden 
    
def init_z(batch_size, G, target_model, min_clip, max_clip, z_dim, target_labels, device, max_iter=10000):
    num_idens = len(target_labels)
    #print('Generating initial points for attacked target classes: Targeted Attack')
    initial_points = {}

    current_iter = 0
    with torch.no_grad():
        while True:
            z = torch.randn(batch_size, z_dim).float().clamp(min=min_clip, max=max_clip).to(device)
            first_img = G(z)
            # our target class is the now the current class of the generated image
            target_classes = decision(first_img, target_model)
            
            for i in range(target_classes.shape[0]):
                current_label = target_classes[i].item()
                if current_label in initial_points or current_label not in target_labels:
                    continue
                
                initial_points[current_label] = z[i]
            
                if len(initial_points) == num_idens:
                    break
            # print("iter {}: current number of distinct labels {}".format(current_iter, len(initial_points)))
            current_iter += 1
            if len(initial_points) == num_idens or current_iter > max_iter:
                break
    
    unmap_labels = []
    for label in target_labels:
        if label not in initial_points.keys():
            unmap_labels.append(label)
    print(f'labels {str(unmap_labels)} can not be generate in iter {max_iter}')
    return initial_points