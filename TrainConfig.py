import os
from easydict import EasyDict
import torch
C = EasyDict()
config = C
cfg = C

C.seed = 2022



""" Training setting """
# trainer
C.num_workers = 6
C.learning_rate = 1e-4
C.weight_decay = 0.98
C.epochs = 100
C.aug_mode = "baseline"
C.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
""" Model setting """
C.drop_out_rate = 0.0
""" Wandb setting """
os.environ['WANDB_API_KEY'] = "55a895793519c48a6e64054c9b396629d3e41d10"
C.project_name = "chache"
C.use_wandb = False
C.resume = False
C.developing = False



