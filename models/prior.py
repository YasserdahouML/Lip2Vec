
import torch
import torch.nn.functional as F
from torch import nn
from util.misc import (accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .transformer import build_transformer
from torch import Tensor
from typing import Iterable, Optional
import numpy as np

from configparser import ConfigParser
import fairseq
from fairseq.modules import SamePad
from av_hubert.avhubert import hubert_pretraining, hubert

import math


def make_conv_pos(e, k, g):
    pos_conv = nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())

    return pos_conv




def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

class MLP(nn.Module):
    """ simple multi-layer perceptron """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
  
                
            
class Prior(nn.Module):
    def __init__(self, transformer, hub_path):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
        """
        super().__init__()
        self.transformer = transformer
        self.down_audio = MLP(1024, 768, 768, 1)
        self.transformer = transformer
        
        
        if hub_path.find('base') >= 0:
            self.hub_dim = 768
        else:
            self.hub_dim = 1024
        
        self.down_video = nn.Linear(self.hub_dim, 768)
        
        self.ups_prior_output = MLP(768, 1024, 1024, 1)

        self.upsample = nn.Sequential(nn.ConvTranspose1d(in_channels=768, out_channels=768, stride=2, kernel_size=5),
                        nn.ReLU())

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([hub_path])
        self.model = models[0]
        
        self.model.load_state_dict(torch.load(hub_path)['model'], strict=False)
        
        self.pos_end = make_conv_pos(768, 128, 16)
            
    def param_grad(self, module, requires_grad):
        for p in module.parameters():
            p.requires_grad_(requires_grad)
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)        
    
    
    
    def forward_test(self, video_src):    
        
        video_reps = self.forward_hubert(video_src)
        
        video_reps = self.down_video(video_reps)
        
        video_reps = self.upsample(video_reps.permute(0,2,1)).permute(0,2,1)

        input_data_pos = self.pos_end(video_reps.transpose(1,2))
        
        video_reps = video_reps.clone() + input_data_pos.transpose(1,2)
        
        hs = self.transformer(video_reps)
        
        hs = self.ups_prior_output(hs)

        return hs
        
        
    def forward_hubert(self, videos):
        
        video_src, video_mask = videos.decompose()
                
        video_reps = self.model.feature_extractor_video(video_src.unsqueeze(1))
        
        fake_audio = video_reps.clone()*0
        
        video_reps = torch.cat([fake_audio, video_reps], dim=1).transpose(1, 2)
        

        video_reps =  self.model.layer_norm(video_reps)
        
        video_reps = self.model.post_extract_proj(video_reps)
                
        video_reps = self.model.encoder(video_reps, video_mask)[0]
        
        return video_reps



def build(args):


    transformer = build_transformer(args)
    
    model = Prior(
        transformer,
        args.hub_path)
    

    return model
