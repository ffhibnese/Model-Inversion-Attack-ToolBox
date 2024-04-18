import torch
import torch.nn.functional as F
from utils.stylegan import create_image
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader


def scores_by_transform(imgs,
                        targets,
                        target_model,
                        transforms,
                        iterations=100):

    score = torch.zeros_like(targets, dtype=torch.float32).to(imgs.device)

    with torch.no_grad():
        for i in range(iterations):
            imgs_transformed = transforms(imgs)
            prediction_vector = target_model(imgs_transformed).softmax(dim=1)
            score += torch.gather(prediction_vector, 1,
                                  targets.unsqueeze(1)).squeeze()
        score = score / iterations
    return score


def perform_final_selection(w,
                            generator,
                            config,
                            targets,
                            target_model,
                            samples_per_target,
                            approach,
                            iterations,
                            batch_size,
                            device,
                            rtpt=None):
    target_values = set(targets.cpu().tolist())
    final_targets = []
    final_w = []
    target_model.eval()

    if approach.strip() == 'transforms':
        transforms = T.Compose([
            T.RandomResizedCrop(size=(224, 224),
                                scale=(0.5, 0.9),
                                ratio=(0.8, 1.2),
                                antialias=True),
            T.RandomHorizontalFlip(0.5)
        ])

    for step, target in enumerate(target_values):
        mask = torch.where(targets == target, True, False).cpu()
        w_masked = w[mask]
        candidates = create_image(w_masked,
                                  generator,
                                  crop_size=config.attack_center_crop,
                                  resize=config.attack_resize,
                                  device=device).cpu()
        targets_masked = targets[mask].cpu()
        scores = []
        dataset = TensorDataset(candidates, targets_masked)
        for imgs, t in DataLoader(dataset, batch_size=batch_size):
            imgs, t = imgs.to(device), t.to(device)

            scores.append(
                scores_by_transform(imgs, t, target_model, transforms,
                                    iterations))
        scores = torch.cat(scores, dim=0).cpu()
        indices = torch.sort(scores, descending=True).indices
        selected_indices = indices[:samples_per_target]
        final_targets.append(targets_masked[selected_indices].cpu())
        final_w.append(w_masked[selected_indices].cpu())

        if rtpt:
            rtpt.step(
                subtitle=f'Sample Selection step {step} of {len(target_values)}'
            )
    final_targets = torch.cat(final_targets, dim=0)
    final_w = torch.cat(final_w, dim=0)
    return final_w, final_targets
