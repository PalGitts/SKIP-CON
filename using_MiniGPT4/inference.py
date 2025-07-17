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
from minigpt4.conversation.conversation import Chat
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='okvqa', help="dataset to evaluate")
args = parser.parse_args()
cfg = Config(args)

print(cfg)

model, vis_processor = init_model(args)
model.eval()
model = model.cuda().half()  # <-- Convert model to float16 (half precision)

print(f'*********************************')


image_path = './metaphor_0.png'  # replace with your image
question = "What the pile of cigerattes signify here in this context?"
image = Image.open(image_path).convert('RGB')
image_tensor = vis_processor(image).unsqueeze(0).cuda().half()  # <-- Convert image to float16
prompt = f"###Human: <Img><ImageHere></Img> {question} ###Assistant:"
texts = [prompt]

with torch.no_grad():
    output = model.generate(image_tensor, texts, max_new_tokens=50)

print(f'I/P prompt: {prompt}')
print("Answer:", output[0])





# python3 inference.py --cfg-path '/home2/palash/p1_Jailbreak/MiniGPT4/eval_configs/minigpt4_llama2_eval.yaml'
# python3 inference.py --cfg-path '/home2/palash/p1_Jailbreak/MiniGPT4/eval_configs/minigpt4_eval.yaml'
# model is set in: \home\u\p1_jb\MiniGPT4\minigpt4\configs\models\minigpt4_llama2.yaml
