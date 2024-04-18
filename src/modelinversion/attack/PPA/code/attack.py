import argparse
import csv
import math
import random
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import wandb
from facenet_pytorch import InceptionResnetV1
from rtpt import RTPT
from torch.utils.data import TensorDataset

from attacks.final_selection import perform_final_selection
from attacks.optimize import Optimization
from datasets.custom_subset import ClassSubset
from metrics.classification_acc import ClassificationAccuracy
from metrics.fid_score import FID_Score
from metrics.prcd import PRCD
from utils.attack_config_parser import AttackConfigParser
from utils.datasets import (create_target_dataset, get_facescrub_idx_to_class,
                            get_stanford_dogs_idx_to_class)
from utils.stylegan import create_image, load_discrimator, load_generator
from utils.wandb import *


def main():
    ####################################
    #        Attack Preparation        #
    ####################################

    # Set devices
    torch.set_num_threads(24)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # Define and parse attack arguments
    parser = create_parser()
    config, args = parse_arguments(parser)

    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load idx to class mappings
    idx_to_class = None
    if config.dataset.lower() == 'facescrub':
        idx_to_class = get_facescrub_idx_to_class()
    elif config.dataset.lower() == 'stanford_dogs':
        idx_to_class = get_stanford_dogs_idx_to_class()
    else:

        class KeyDict(dict):

            def __missing__(self, key):
                return key

        idx_to_class = KeyDict()

    # Load pre-trained StyleGan2 components
    G = load_generator(config.stylegan_model)
    D = load_discrimator(config.stylegan_model)
    num_ws = G.num_ws

    # Load target model and set dataset
    target_model = config.create_target_model()
    target_model_name = target_model.name
    target_dataset = config.get_target_dataset()

    # Distribute models
    target_model = torch.nn.DataParallel(target_model, device_ids=gpu_devices)
    target_model.name = target_model_name
    synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    synthesis.num_ws = num_ws
    discriminator = torch.nn.DataParallel(D, device_ids=gpu_devices)

    # Load basic attack parameters
    num_epochs = config.attack['num_epochs']
    batch_size_single = config.attack['batch_size']
    batch_size = config.attack['batch_size'] * torch.cuda.device_count()
    targets = config.create_target_vector()

    # Create initial style vectors
    w, w_init, x, V = create_initial_vectors(config, G, target_model, targets,
                                             device)
    del G

    # Initialize wandb logging
    if config.logging:
        optimizer = config.create_optimizer(params=[w])
        wandb_run = init_wandb_logging(optimizer, target_model_name, config,
                                       args)
        run_id = wandb_run.id

    # Print attack configuration
    print(
        f'Start attack against {target_model.name} optimizing w with shape {list(w.shape)} ',
        f'and targets {dict(Counter(targets.cpu().numpy()))}.')
    print(f'\nAttack parameters')
    for key in config.attack:
        print(f'\t{key}: {config.attack[key]}')
    print(
        f'Performing attack on {torch.cuda.device_count()} gpus and an effective batch size of {batch_size} images.'
    )

    # Initialize RTPT
    rtpt = None
    if args.rtpt:
        max_iterations = math.ceil(w.shape[0] / batch_size) \
            + int(math.ceil(w.shape[0] / (batch_size * 3))) \
            + 2 * int(math.ceil(config.final_selection['samples_per_target'] * len(set(targets.cpu().tolist())) / (batch_size * 3))) \
            + 2 * len(set(targets.cpu().tolist()))
        rtpt = RTPT(name_initials='LS',
                    experiment_name='Model_Inversion',
                    max_iterations=max_iterations)
        rtpt.start()

    # Log initial vectors
    if config.logging:
        Path("results").mkdir(parents=True, exist_ok=True)
        init_w_path = f"results/init_w_{run_id}.pt"
        torch.save(w.detach(), init_w_path)
        wandb.save(init_w_path)

    # Create attack transformations
    attack_transformations = config.create_attack_transformations()

    ####################################
    #         Attack Iteration         #
    ####################################
    optimization = Optimization(target_model, synthesis, discriminator,
                                attack_transformations, num_ws, config)

    # Collect results
    w_optimized = []

    # Prepare batches for attack
    for i in range(math.ceil(w.shape[0] / batch_size)):
        w_batch = w[i * batch_size:(i + 1) * batch_size].cuda()
        targets_batch = targets[i * batch_size:(i + 1) * batch_size].cuda()
        print(
            f'\nOptimizing batch {i+1} of {math.ceil(w.shape[0] / batch_size)} targeting classes {set(targets_batch.cpu().tolist())}.'
        )

        # Run attack iteration
        torch.cuda.empty_cache()
        w_batch_optimized = optimization.optimize(w_batch, targets_batch,
                                                  num_epochs).detach().cpu()

        if rtpt:
            num_batches = math.ceil(w.shape[0] / batch_size)
            rtpt.step(subtitle=f'batch {i+1} of {num_batches}')

        # Collect optimized style vectors
        w_optimized.append(w_batch_optimized)

    # Concatenate optimized style vectors
    w_optimized_unselected = torch.cat(w_optimized, dim=0)
    torch.cuda.empty_cache()
    del discriminator

    # Log optimized vectors
    if config.logging:
        optimized_w_path = f"results/optimized_w_{run_id}.pt"
        torch.save(w_optimized_unselected.detach(), optimized_w_path)
        wandb.save(optimized_w_path)

    ####################################
    #          Filter Results          #
    ####################################

    # Filter results
    if config.final_selection:
        print(
            f'\nSelect final set of max. {config.final_selection["samples_per_target"]} ',
            f'images per target using {config.final_selection["approach"]} approach.'
        )
        final_w, final_targets = perform_final_selection(
            w_optimized_unselected,
            synthesis,
            config,
            targets,
            target_model,
            device=device,
            batch_size=batch_size * 10,
            **config.final_selection,
            rtpt=rtpt)
        print(f'Selected a total of {final_w.shape[0]} final images ',
              f'of target classes {set(final_targets.cpu().tolist())}.')
    else:
        final_targets, final_w = targets, w_optimized_unselected
    del target_model

    # Log selected vectors
    if config.logging:
        optimized_w_path_selected = f"results/optimized_w_selected_{run_id}.pt"
        torch.save(final_w.detach(), optimized_w_path_selected)
        wandb.save(optimized_w_path_selected)
        wandb.config.update({'w_path': optimized_w_path})

    ####################################
    #         Attack Accuracy          #
    ####################################

    # Compute attack accuracy with evaluation model on all generated samples
    try:
        evaluation_model = config.create_evaluation_model()
        evaluation_model = torch.nn.DataParallel(evaluation_model)
        evaluation_model.to(device)
        evaluation_model.eval()
        class_acc_evaluator = ClassificationAccuracy(evaluation_model,
                                                     device=device)

        acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
            w_optimized_unselected,
            targets,
            synthesis,
            config,
            batch_size=batch_size * 2,
            resize=299,
            rtpt=rtpt)

        if config.logging:
            try:
                filename_precision = write_precision_list(
                    f'results/precision_list_unfiltered_{run_id}',
                    precision_list)
                wandb.save(filename_precision)
            except:
                pass
        print(
            f'\nUnfiltered Evaluation of {final_w.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}',
            f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
        )

        # Compute attack accuracy on filtered samples
        if config.final_selection:
            acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
                final_w,
                final_targets,
                synthesis,
                config,
                batch_size=batch_size * 2,
                resize=299,
                rtpt=rtpt)
            if config.logging:
                filename_precision = write_precision_list(
                    f'results/precision_list_filtered_{run_id}',
                    precision_list)
                wandb.save(filename_precision)

            print(
                f'Filtered Evaluation of {final_w.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}, ',
                f'accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
            )
        del evaluation_model

    except Exception:
        print(traceback.format_exc())

    ####################################
    #    FID Score and GAN Metrics     #
    ####################################

    fid_score = None
    precision, recall = None, None
    density, coverage = None, None
    try:
        # set transformations
        crop_size = config.attack_center_crop
        target_transform = T.Compose([
            T.ToTensor(),
            T.Resize((299, 299), antialias=True),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # create datasets
        attack_dataset = TensorDataset(final_w, final_targets)
        attack_dataset.targets = final_targets
        training_dataset = create_target_dataset(target_dataset,
                                                 target_transform)
        training_dataset = ClassSubset(
            training_dataset,
            target_classes=torch.unique(final_targets).cpu().tolist())

        # compute FID score
        fid_evaluation = FID_Score(training_dataset,
                                   attack_dataset,
                                   device=device,
                                   crop_size=crop_size,
                                   generator=synthesis,
                                   batch_size=batch_size * 3,
                                   dims=2048,
                                   num_workers=8,
                                   gpu_devices=gpu_devices)
        fid_score = fid_evaluation.compute_fid(rtpt)
        print(
            f'FID score computed on {final_w.shape[0]} attack samples and {config.dataset}: {fid_score:.4f}'
        )

        # compute precision, recall, density, coverage
        prdc = PRCD(training_dataset,
                    attack_dataset,
                    device=device,
                    crop_size=crop_size,
                    generator=synthesis,
                    batch_size=batch_size * 3,
                    dims=2048,
                    num_workers=8,
                    gpu_devices=gpu_devices)
        precision, recall, density, coverage = prdc.compute_metric(
            num_classes=config.num_classes, k=3, rtpt=rtpt)
        print(
            f' Precision: {precision:.4f}, Recall: {recall:.4f}, Density: {density:.4f}, Coverage: {coverage:.4f}'
        )

    except Exception:
        print(traceback.format_exc())

    ####################################
    #         Feature Distance         #
    ####################################
    avg_dist_inception = None
    avg_dist_facenet = None
    try:
        # Load Inception-v3 evaluation model and remove final layer
        evaluation_model_dist = config.create_evaluation_model()
        evaluation_model_dist.model.fc = torch.nn.Sequential()
        evaluation_model_dist = torch.nn.DataParallel(evaluation_model_dist,
                                                      device_ids=gpu_devices)
        evaluation_model_dist.to(device)
        evaluation_model_dist.eval()

        # Compute average feature distance on Inception-v3
        evaluate_inception = DistanceEvaluation(evaluation_model_dist,
                                                synthesis, 299,
                                                config.attack_center_crop,
                                                target_dataset, config.seed)
        avg_dist_inception, mean_distances_list = evaluate_inception.compute_dist(
            final_w,
            final_targets,
            batch_size=batch_size_single * 5,
            rtpt=rtpt)

        if config.logging:
            try:
                filename_distance = write_precision_list(
                    f'results/distance_inceptionv3_list_filtered_{run_id}',
                    mean_distances_list)
                wandb.save(filename_distance)
            except:
                pass

        print('Mean Distance on Inception-v3: ',
              avg_dist_inception.cpu().item())
        # Compute feature distance only for facial images
        if target_dataset in [
                'facescrub', 'celeba_identities', 'celeba_attributes'
        ]:
            # Load FaceNet model for face recognition
            facenet = InceptionResnetV1(pretrained='vggface2')
            facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
            facenet.to(device)
            facenet.eval()

            # Compute average feature distance on facenet
            evaluater_facenet = DistanceEvaluation(facenet, synthesis, 160,
                                                   config.attack_center_crop,
                                                   target_dataset, config.seed)
            avg_dist_facenet, mean_distances_list = evaluater_facenet.compute_dist(
                final_w,
                final_targets,
                batch_size=batch_size_single * 8,
                rtpt=rtpt)
            if config.logging:
                filename_distance = write_precision_list(
                    f'results/distance_facenet_list_filtered_{run_id}',
                    mean_distances_list)
                wandb.save(filename_distance)

            print('Mean Distance on FaceNet: ', avg_dist_facenet.cpu().item())
    except Exception:
        print(traceback.format_exc())

    ####################################
    #          Finish Logging          #
    ####################################

    if rtpt:
        rtpt.step(subtitle=f'Finishing up')

    # Logging of final results
    if config.logging:
        print('Finishing attack, logging results and creating sample images.')
        num_classes = 10
        num_imgs = 8
        # Sample final images from the first and last classes
        label_subset = set(
            list(set(targets.tolist()))[:int(num_classes / 2)] +
            list(set(targets.tolist()))[-int(num_classes / 2):])
        log_imgs = []
        log_targets = []
        log_predictions = []
        log_max_confidences = []
        log_target_confidences = []
        # Log images with smallest feature distance
        for label in label_subset:
            mask = torch.where(final_targets == label, True, False)
            w_masked = final_w[mask][:num_imgs]
            imgs = create_image(w_masked,
                                synthesis,
                                crop_size=config.attack_center_crop,
                                resize=config.attack_resize)
            log_imgs.append(imgs)
            log_targets += [label for i in range(num_imgs)]
            log_predictions.append(torch.tensor(predictions)[mask][:num_imgs])
            log_max_confidences.append(
                torch.tensor(maximum_confidences)[mask][:num_imgs])
            log_target_confidences.append(
                torch.tensor(target_confidences)[mask][:num_imgs])

        log_imgs = torch.cat(log_imgs, dim=0)
        log_predictions = torch.cat(log_predictions, dim=0)
        log_max_confidences = torch.cat(log_max_confidences, dim=0)
        log_target_confidences = torch.cat(log_target_confidences, dim=0)

        log_final_images(log_imgs, log_predictions, log_max_confidences,
                         log_target_confidences, idx_to_class)

        # Find closest training samples to final results
        log_nearest_neighbors(log_imgs,
                              log_targets,
                              evaluation_model_dist,
                              'InceptionV3',
                              target_dataset,
                              img_size=299,
                              seed=config.seed)

        # Use FaceNet only for facial images
        facenet = InceptionResnetV1(pretrained='vggface2')
        facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
        facenet.to(device)
        facenet.eval()
        if target_dataset in [
                'facescrub', 'celeba_identities', 'celeba_attributes'
        ]:
            log_nearest_neighbors(log_imgs,
                                  log_targets,
                                  facenet,
                                  'FaceNet',
                                  target_dataset,
                                  img_size=160,
                                  seed=config.seed)

        # Final logging
        final_wandb_logging(avg_correct_conf, avg_total_conf, acc_top1,
                            acc_top5, avg_dist_facenet, avg_dist_inception,
                            fid_score, precision, recall, density, coverage)


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing model inversion attack')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    parser.add_argument('--no_rtpt',
                        action='store_false',
                        dest="rtpt",
                        help='Disable RTPT')
    return parser


def parse_arguments(parser):
    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load attack config
    config = AttackConfigParser(args.config)

    return config, args


def create_initial_vectors(config, G, target_model, targets, device):
    with torch.no_grad():
        w = config.create_candidates(G, target_model, targets).cpu()
        if config.attack['single_w']:
            w = w[:, 0].unsqueeze(1)
        w_init = deepcopy(w)
        x = None
        V = None
    return w, w_init, x, V


def write_precision_list(filename, precision_list):
    filename = f"{filename}.csv"
    with open(filename, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for row in precision_list:
            wr.writerow(row)
    return filename


def log_attack_progress(loss,
                        target_loss,
                        discriminator_loss,
                        discriminator_weight,
                        mean_conf,
                        lr,
                        imgs=None,
                        captions=None):
    if imgs is not None:
        imgs = [
            wandb.Image(img.permute(1, 2, 0).numpy(), caption=caption)
            for img, caption in zip(imgs, captions)
        ]
        wandb.log({
            'total_loss': loss,
            'target_loss': target_loss,
            'discriminator_loss': discriminator_loss,
            'discriminator_weight': discriminator_weight,
            'mean_conf': mean_conf,
            'learning_rate': lr,
            'samples': imgs
        })
    else:
        wandb.log({
            'total_loss': loss,
            'target_loss': target_loss,
            'discriminator_loss': discriminator_loss,
            'discriminator_weight': discriminator_weight,
            'mean_conf': mean_conf,
            'learning_rate': lr
        })


def init_wandb_logging(optimizer, target_model_name, config, args):
    lr = optimizer.param_groups[0]['lr']
    optimizer_name = type(optimizer).__name__
    if not 'name' in config.wandb['wandb_init_args']:
        config.wandb['wandb_init_args'][
            'name'] = f'{optimizer_name}_{lr}_{target_model_name}'
    wandb_config = config.create_wandb_config()
    run = wandb.init(config=wandb_config, **config.wandb['wandb_init_args'])
    wandb.save(args.config)
    return run


def intermediate_wandb_logging(optimizer, targets, confidences, loss,
                               target_loss, discriminator_loss,
                               discriminator_weight, mean_conf, imgs, idx2cls):
    lr = optimizer.param_groups[0]['lr']
    target_classes = [idx2cls[idx.item()] for idx in targets.cpu()]
    conf_list = [conf.item() for conf in confidences]
    if imgs is not None:
        img_captions = [
            f'{target} ({conf:.4f})'
            for target, conf in zip(target_classes, conf_list)
        ]
        log_attack_progress(loss,
                            target_loss,
                            discriminator_loss,
                            discriminator_weight,
                            mean_conf,
                            lr,
                            imgs,
                            captions=img_captions)
    else:
        log_attack_progress(loss, target_loss, discriminator_loss,
                            discriminator_weight, mean_conf, lr)


def log_nearest_neighbors(imgs, targets, eval_model, model_name, dataset,
                          img_size, seed):
    # Find closest training samples to final results
    evaluater = DistanceEvaluation(eval_model, None, img_size, None, dataset,
                                   seed)
    closest_samples, distances = evaluater.find_closest_training_sample(
        imgs, targets)
    closest_samples = [
        wandb.Image(img.permute(1, 2, 0).cpu().numpy(),
                    caption=f'distance={d:.4f}')
        for img, d in zip(closest_samples, distances)
    ]
    wandb.log({f'closest_samples {model_name}': closest_samples})


def log_final_images(imgs, predictions, max_confidences, target_confidences,
                     idx2cls):
    wand_imgs = [
        wandb.Image(
            img.permute(1, 2, 0).numpy(),
            caption=
            f'pred={idx2cls[pred.item()]} ({max_conf:.2f}), target_conf={target_conf:.2f}'
        ) for img, pred, max_conf, target_conf in zip(
            imgs.cpu(), predictions, max_confidences, target_confidences)
    ]
    wandb.log({'final_images': wand_imgs})


def final_wandb_logging(avg_correct_conf, avg_total_conf, acc_top1, acc_top5,
                        avg_dist_facenet, avg_dist_eval, fid_score, precision,
                        recall, density, coverage):
    wandb.save('attacks/gradient_based.py')
    wandb.run.summary['correct_avg_conf'] = avg_correct_conf
    wandb.run.summary['total_avg_conf'] = avg_total_conf
    wandb.run.summary['evaluation_acc@1'] = acc_top1
    wandb.run.summary['evaluation_acc@5'] = acc_top5
    wandb.run.summary['avg_dist_facenet'] = avg_dist_facenet
    wandb.run.summary['avg_dist_evaluation'] = avg_dist_eval
    wandb.run.summary['fid_score'] = fid_score
    wandb.run.summary['precision'] = precision
    wandb.run.summary['recall'] = recall
    wandb.run.summary['density'] = density
    wandb.run.summary['coverage'] = coverage

    wandb.finish()


if __name__ == '__main__':
    main()
