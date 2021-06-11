# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, base_encoder, num_classes, norm_layer=None):
        super(ResNet, self).__init__()
        self.backbone = base_encoder(norm_layer=norm_layer)
        assert not hasattr(self.backbone, 'fc'), "fc should not in backbone"
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class FixMatch(nn.Module):

    def __init__(self, base_encoder, num_classes=1000, eman=False, momentum=0.999, norm=None):
        super(FixMatch, self).__init__()
        self.eman = eman
        self.momentum = momentum
        self.main = ResNet(base_encoder, num_classes, norm_layer=norm)
        # build ema model
        if eman:
            print("using EMAN as techer model")
            self.ema = ResNet(base_encoder, num_classes, norm_layer=norm)
            for param_main, param_ema in zip(self.main.parameters(), self.ema.parameters()):
                param_ema.data.copy_(param_main.data)  # initialize
                param_ema.requires_grad = False  # not update by gradient
        else:
            self.ema = None

    def momentum_update_ema(self):
        state_dict_main = self.main.state_dict()
        state_dict_ema = self.ema.state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, "state_dict names are different!"
            assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_ema:
                v_ema.copy_(v_main)
            else:
                v_ema.copy_(v_ema * self.momentum + (1. - self.momentum) * v_main)

    def forward(self, im_x, im_u_w=None, im_u_s=None):
        if im_u_w is None and im_u_s is None:
            logits = self.main(im_x)
            return logits

        batch_size_x = im_x.shape[0]
        if not self.eman:
            inputs = torch.cat((im_x, im_u_w, im_u_s))
            logits = self.main(inputs)
            logits_x = logits[:batch_size_x]
            logits_u_w, logits_u_s = logits[batch_size_x:].chunk(2)
        else:
            # use ema model for pesudo labels
            inputs = torch.cat((im_x, im_u_s))
            logits = self.main(inputs)
            logits_x = logits[:batch_size_x]
            logits_u_s = logits[batch_size_x:]
            with torch.no_grad():  # no gradient to ema model
                logits_u_w = self.ema(im_u_w)

        return logits_x, logits_u_w, logits_u_s


def get_fixmatch_model(model):
    """
    Args:
        model (str or callable):

    Returns:
        FixMatch model
    """
    if isinstance(model, str):
        model = {
            "FixMatch": FixMatch,
        }[model]
    return model
