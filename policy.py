import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

import sys
sys.path.append('../v2r/Video2Reward/')     #TODO: change this to the path of the Video2Reward folder
import simple_reward_model as v2r_model

class ACTPolicy(nn.Module):
    def __init__(self, args_override, reward_model='ours' , weighted=False):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

        self.weighted = weighted
        self.reward_model = reward_model

        if self.weighted:
            self.v2r_reward_model = v2r_model.Model(model_type="resnet34")
            self.v2r_reward_model.to(model.device)
            self.v2r_reward_model.load_state_dict(torch.load('../model_499.pth'))  #TODO: change this to the path of the model
            self.v2r_reward_model.eval()

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])
        v2r_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(mean=mean, std=std)
        ])
        image = v2r_transform(image.squeeze(0)).unsqueeze(0)
        
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            actions = actions.to(torch.float32)
            qpos = qpos.to(torch.float32)

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()

            if self.weighted:
                assert init_frame is not None and final_frame is not None
                init_frame = v2r_transform(init_frame)
                final_frame = v2r_transform(final_frame)

                triplate = torch.cat([init_frame, image, final_frame]).unsqueeze(0).to(self.v2r_reward_model.device).float()
                weighting_reward = self.v2r_reward_model.compute_reward(triplate)

            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1 * weighting_reward if self.weighted else l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override, reward_model='ours' , weighted=False):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer
        self.weighted = weighted
        self.reward_model = reward_model
        
        if self.weighted:
            self.v2r_reward_model = v2r_model.Model(model_type="resnet34")
            self.v2r_reward_model.to(model.device)
            self.v2r_reward_model.load_state_dict(torch.load('../model_499.pth'))  #TODO: change this to the path of the model
            self.v2r_reward_model.eval()

    def __call__(self, qpos, image, actions=None, is_pad=None, init_frame=None, final_frame=None):
        env_state = None # TODO
        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])
        v2r_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(mean=mean, std=std)
        ])
        image = v2r_transform(image)

        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)

            if self.weighted:
                assert init_frame is not None and final_frame is not None
                init_frame = v2r_transform(init_frame)
                final_frame = v2r_transform(final_frame)

                triplate = torch.cat([init_frame, image, final_frame]).unsqueeze(0).to(self.v2r_reward_model.device).float()
                weighting_reward = self.v2r_reward_model.compute_reward(triplate)

            loss_dict = dict()
            loss_dict['mse'] = mse * weighting_reward if self.weighted else mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
