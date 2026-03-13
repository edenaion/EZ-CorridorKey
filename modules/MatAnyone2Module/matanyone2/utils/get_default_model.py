"""
A helper function to get a default model for quick testing
"""
from omegaconf import open_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

import torch
from matanyone2.model.matanyone2 import MatAnyone2

def get_matanyone2_model(ckpt_path, device=None) -> MatAnyone2:
    GlobalHydra.instance().clear()
    initialize(version_base='1.3.2', config_path="../config", job_name="eval_our_config")
    cfg = compose(config_name="eval_matanyone_config")
    
    with open_dict(cfg):
        cfg['weights'] = ckpt_path

    # Load the network weights
    if device is not None:
        matanyone2 = MatAnyone2(cfg, single_object=True).to(device).eval()
        model_weights = torch.load(cfg.weights, map_location=device)
    else:  # if device is not specified, `.cuda()` by default
        matanyone2 = MatAnyone2(cfg, single_object=True).cuda().eval()
        model_weights = torch.load(cfg.weights)
        
    matanyone2.load_weights(model_weights)

    return matanyone2
