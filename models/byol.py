# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from math import cos, pi
import torch
import torch.nn as nn
from utils import init


class MLP(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """
    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=2):
        super(MLP, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp-1):
            mlps.append(nn.Linear(in_channels, hid_channels, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Linear(hid_channels, out_channels, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear='normal'):
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x


class Encoder(nn.Module):
    def __init__(self, base_encoder, hid_dim, out_dim, norm_layer=None, num_mlp=2):
        super(Encoder, self).__init__()
        self.backbone = base_encoder(norm_layer=norm_layer)
        in_dim = self.backbone.out_channels
        self.neck = MLP(in_dim, hid_dim, out_dim, norm_layer=norm_layer, num_mlp=num_mlp)
        self.neck.init_weights(init_linear='kaiming')

    def forward(self, im):
        out = self.backbone(im)
        out = self.neck(out)
        return out


class BYOL(nn.Module):
    """
    BYOL re-implementation. Paper: https://arxiv.org/abs/2006.07733
    """
    def __init__(self, base_encoder, dim=256, m=0.996, hid_dim=4096, norm_layer=None, num_neck_mlp=2):
        super(BYOL, self).__init__()

        self.base_m = m
        self.curr_m = m

        # create the encoders
        # num_classes is the output fc dimension
        self.online_net = Encoder(base_encoder, hid_dim, dim, norm_layer, num_neck_mlp)
        self.target_net = Encoder(base_encoder, hid_dim, dim, norm_layer, num_neck_mlp)
        self.predictor = MLP(dim, hid_dim, dim, norm_layer=norm_layer)
        self.predictor.init_weights()

        # copy params from online model to target model
        for param_ol, param_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)  # initialize
            param_tgt.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def momentum_update(self, cur_iter, max_iter):
        """
        Momentum update of the target network.
        """
        # momentum anneling
        momentum = 1. - (1. - self.base_m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2.0
        self.curr_m = momentum
        # parameter update for target network
        for param_ol, param_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
            param_tgt.data = param_tgt.data * momentum + param_ol.data * (1. - momentum)

    def loss_func(self, pred, target):
        """
        Args:
            pred (Tensor): NxC input features.
            target (Tensor): NxC target features.
        """
        N = pred.size(0)
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = 2 - 2 * (pred_norm * target_norm).sum() / N
        return loss

    def forward(self, im_v1, im_v2=None):
        """
        Input:
            im_v1: a batch of view1 images
            im_v2: a batch of view2 images
        Output:
            loss
        """
        # for inference, online_net.backbone model only
        if im_v2 is None:
            feats = self.online_net.backbone(im_v1)
            return feats

        # compute online_net features
        proj_online_v1 = self.online_net(im_v1)
        proj_online_v2 = self.online_net(im_v2)

        # compute target_net features
        with torch.no_grad():  # no gradient to keys
            proj_target_v1 = self.target_net(im_v1).clone().detach()
            proj_target_v2 = self.target_net(im_v2).clone().detach()

        # loss
        loss = self.loss_func(self.predictor(proj_online_v1), proj_target_v2) + \
            self.loss_func(self.predictor(proj_online_v2), proj_target_v1)

        return loss


class BYOLEMAN(BYOL):

    def __init__(self, base_encoder, dim=256, m=0.996, hid_dim=4096, norm_layer=None, num_neck_mlp=2):
        super(BYOL, self).__init__()

        self.base_m = m
        self.curr_m = m

        # create the encoders
        # num_classes is the output fc dimension
        self.online_net = Encoder(base_encoder, hid_dim, dim, norm_layer, num_neck_mlp)
        self.target_net = Encoder(base_encoder, hid_dim, dim, num_mlp=num_neck_mlp)
        self.predictor = MLP(dim, hid_dim, dim, norm_layer=norm_layer)
        self.predictor.init_weights()

        # copy params from online model to target model
        for param_ol, param_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)  # initialize
            param_tgt.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def momentum_update(self, cur_iter, max_iter):
        """
        Momentum update of the target network.
        """
        # momentum anneling
        momentum = 1. - (1. - self.base_m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2.0
        self.curr_m = momentum
        # parameter update for target network
        state_dict_ol = self.online_net.state_dict()
        state_dict_tgt = self.target_net.state_dict()
        for (k_ol, v_ol), (k_tgt, v_tgt) in zip(state_dict_ol.items(), state_dict_tgt.items()):
            assert k_tgt == k_ol, "state_dict names are different!"
            assert v_ol.shape == v_tgt.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_tgt:
                v_tgt.copy_(v_ol)
            else:
                v_tgt.copy_(v_tgt * momentum + (1. - momentum) * v_ol)


def get_byol_model(model):
    """
    Args:
        model (str or callable):

    Returns:
        BYOL model
    """
    if isinstance(model, str):
        model = {
            "BYOL": BYOL,
            "BYOLEMAN": BYOLEMAN,
        }[model]
    return model
