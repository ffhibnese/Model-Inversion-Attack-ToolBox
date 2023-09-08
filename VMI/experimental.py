import os
import torch
from types import SimpleNamespace
import json
import yaml

# Local
import data
from model_utils import instantiate_generator, instantiate_discriminator, load_cls_embed, load_cls_z_to_lsm, TargetDatasetReconstructionEvaluation
from eval_pretrained_face_classifier import PretrainedInsightFaceClassifier, FinetunednsightFaceClassifier
from classify_mnist import get_model, CelebAPretrained


def load_gan_from_ckpt(ckpt_path, device):
    with open(os.path.join(os.path.split(ckpt_path)[0], 'args.json'), 'r') as f:
        model_args = json.load(f)
        model_args = SimpleNamespace(**model_args)
        if 'nc' not in dir(model_args):
            model_args.nc = 3
        if model_args.model == 'kplus1gan' or ('lambda_diversity' in dir(model_args) and model_args.lambda_diversity) > 0:
            model_args.cdim = 50
        elif model_args.model == 'dcgan_aux':
            raise
        else:
            model_args.cdim = 1

    generator = instantiate_generator(model_args, device)
    loaded = torch.load(os.path.join(ckpt_path))
    generator.load_state_dict(loaded)
    generator.eval()
    return generator


def load_gan_from_config(config_fname, device):
    config = yaml.load(open(f'configs/{config_fname}', 'r'))
    config['prior_gan']['g_path'] = os.path.join(
        os.environ['ROOT1'], config['prior_gan']['g_path'])
    generator = load_gan_from_ckpt(config['prior_gan']['g_path'], device)
    return generator


class AttackExperiment:
    def __init__(self, config_fname, device, db=0, fixed_id=-1, run_target_feat_eval=False, k=5):
        self.config_fname = config_fname
        if config_fname.startswith('/'):
            self.config = yaml.load(open(config_fname, 'r'))
        else:
            self.config = yaml.load(open(f'configs/{config_fname}', 'r'))
        if db and 'celeba' in self.config['data']['aux']:
            self.config['data']['aux'] = self.config['data']['target'] = 'celeba-db'
        # Check exp type
        if 'exp_type' in self.config:
            experiment_type = self.config['exp_type']
        else:
            experiment_type = 'mi-attack'
        # # Prepend ROOT1 to paths
        # if 'prior_gan' in self.config and not self.config['prior_gan']['g_path'].startswith('/'):
        #     self.config['prior_gan']['g_path'] = os.path.join(
        #         os.environ['ROOT1'], self.config['prior_gan']['g_path'])
        #     self.config['prior_gan']['d_path'] = os.path.join(
        #         os.environ['ROOT1'], self.config['prior_gan']['d_path'])
        # if 'target_cls' in self.config and not self.config['target_cls']['path'].startswith('/'):
        #     self.config['target_cls']['path'] = os.path.join(
        #         os.environ['ROOT1'], self.config['target_cls']['path'])
        # if 'evaluation_cls' in self.config and not self.config['evaluation_cls']['path'].startswith('/'):
        #     self.config['evaluation_cls']['path'] = os.path.join(
        #         os.environ['ROOT1'], self.config['evaluation_cls']['path'])

        # Load data
        if 'image_size' not in self.config['data']:
            self.config['data']['image_size'] = 64  # default

        self.dat = data.load_data(
            self.config['data']['aux'], imgsize=self.config['data']['image_size'], device=device)
        self.target_dataset = data.load_data(
            self.config['data']['target'], imgsize=self.config['data']['image_size'], device=device)
        self.nc = self.dat['nc']

        # Load target cls
        if experiment_type == 'fsg':
            if self.config['target_cls']['name'] != 'none':
                with open(os.path.join(os.path.split(self.config['target_cls']['path'])[0], 'args.json'), 'r') as f:
                    ecls_args = json.load(f)
                    ecls_args = SimpleNamespace(**ecls_args)
                self.target_classifier = get_model(ecls_args, device)[0]
                if self.config['target_cls']['path']:
                    loaded = torch.load(self.config['target_cls']['path'])
                    self.target_classifier.load_state_dict(loaded)
                self.target_classifier.eval()

        else:
            if 'target_cls' in self.config and self.config['target_cls']['name'] != 'none':
                if self.config['target_cls']['name'] == 'CelebAPretrained':
                    target_classifier = CelebAPretrained()
                    self.target_extract_feat = target_classifier.embed_img
                    self.target_logsoftmax = target_classifier.forward
                    self.target_logits = target_classifier.logits
                    self.target_z_to_lsm = target_classifier.z_to_lsm
                    self.cdim = 0  # dummy
                else:
                    self.target_extract_feat = load_cls_embed(
                        self.config['data']['target'], self.config['target_cls']['path'], device)
                    self.target_logsoftmax = load_cls_embed(
                        self.config['data']['target'], self.config['target_cls']['path'], device, classify=True)
                    self.target_logits = load_cls_embed(
                        self.config['data']['target'], self.config['target_cls']['path'], device, classify=False, logits=True)
                    self.target_z_to_lsm = load_cls_z_to_lsm(
                        self.config['data']['target'], self.config['target_cls']['path'], device)
                    self.cdim = self.target_extract_feat(
                        self.target_dataset["X_train"][:2].cuda() / 2 + 0.5).shape[-1]

        # Load eval cls
        if 'evaluation_cls' in self.config:
            if self.config['evaluation_cls']['class'] == 'PretrainedInsightFaceClassifier':
                evaluation_classifier = PretrainedInsightFaceClassifier(
                    'cuda:0', pad=True)
                self.evaluation_classifier = evaluation_classifier
                bgr = True
            elif self.config['evaluation_cls']['class'] == 'FinetunednsightFaceClassifier':
                evaluation_classifier = FinetunednsightFaceClassifier(
                    'cuda:0', 1, eval_mode=True, pad=True)
                evaluation_classifier.load_state_dict(
                    torch.load(self.config['evaluation_cls']['path']))
                evaluation_classifier.eval()
                evaluation_classifier.cuda()
                self.evaluation_classifier = evaluation_classifier
                bgr = True
            else:
                if self.config['evaluation_cls']['class'] != 'none':
                    with open(os.path.join(os.path.split(self.config['evaluation_cls']['path'])[0], 'args.json'), 'r') as f:
                        ecls_args = json.load(f)
                        ecls_args = SimpleNamespace(**ecls_args)
                    self.evaluation_classifier = get_model(
                        ecls_args, device)[0]
                    loaded = torch.load(self.config['evaluation_cls']['path'])
                    self.evaluation_classifier.load_state_dict(loaded)
                    self.evaluation_classifier.eval()
                    bgr = False

            if fixed_id > -1:
                idxs = self.target_dataset['Y_train'] == fixed_id
                train_x = self.target_dataset['X_train'][idxs].cuda()
                train_y = self.target_dataset['Y_train'][idxs].cuda()
                idxs = self.target_dataset['Y_test'] == fixed_id
                test_x = self.target_dataset['X_test'][idxs].cuda()
                test_y = self.target_dataset['Y_test'][idxs].cuda()
                target_x = torch.cat([train_x, test_x])
                target_y = torch.cat([train_y, test_y])
            else:
                target_x = self.target_dataset['X_test'].cuda()
                target_y = self.target_dataset['Y_test'].cuda()
            # Use Max 3000 images (otherwise computing manifold doesn't fit int mem)
            self.target_x = target_x[:3000]
            self.target_y = target_y[:3000]
            self.target_eval_runner = TargetDatasetReconstructionEvaluation(self.evaluation_classifier, self.target_x, self.target_y,
                                                                            bgr=bgr,
                                                                            run_target_feat_eval=run_target_feat_eval,
                                                                            k=k)

        # Load Prior G
        if 'prior_gan' in self.config:
            with open(os.path.join(os.path.split(self.config['prior_gan']['g_path'])[0], 'args.json'), 'r') as f:
                model_args = json.load(f)
                model_args = SimpleNamespace(**model_args)
                model_args.nc = self.dat['nc']
                if model_args.model == 'kplus1gan' or ('lambda_diversity' in dir(model_args) and model_args.lambda_diversity) > 0:
                    model_args.cdim = 50
                elif model_args.model == 'dcgan_aux':
                    model_args.cdim = self.target_dataset['nclass']
                    self.gan_method = 'dcgan_aux'
                else:
                    model_args.cdim = 1
                    self.gan_method = 'dcgan'

            generator = instantiate_generator(model_args, device)
            loaded = torch.load(os.path.join(
                self.config['prior_gan']['g_path']))
            generator.load_state_dict(loaded)
            generator.eval()
            # assert not generator.is_conditional

            # discriminator = instantiate_discriminator(model_args, None, device)
            # loaded = torch.load(self.config['prior_gan']['d_path'])
            # if 'index2class' in loaded:
            #     loaded.pop('index2class')
            # discriminator.load_state_dict(loaded)
            # discriminator.eval()
            # assert discriminator.use_sigmoid
            # self.gan_args = model_args
            # self.generator = generator
            # self.discriminator = discriminator


if __name__ == '__main__':
    exp = AttackExperiment('dev.yaml', 'cuda:0', db=1)

    import ipdb
    ipdb.set_trace()
