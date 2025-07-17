import sys
path = "/home2/palash/p1_Jailbreak/MiniGPT4/common"
sys.path.append(path)

import os
import re
import json
import argparse
from collections import defaultdict
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
# from minigpt4.conversation.conversation import CONV_VISION_minigptv2
# from minigpt4.datasets.datasets.coco_caption import RefCOCOEvalData


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='okvqa', help="dataset to evaluate")
args = parser.parse_args()
cfg = Config(args)

print(cfg)
model, vis_processor = init_model(args)
model.eval()
print(f'*********************************')


# python3 inference.py --cfg-path '/home2/palash/p1_Jailbreak/MiniGPT4/eval_configs/minigpt4_llama2_eval.yaml'