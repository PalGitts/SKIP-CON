import sys
path = "/home2/palash/p1_Jailbreak/MiniGPT4/common"
sys.path.append(path)

import os, re, gc, json, argparse, random, torch
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from minigpt4.conversation.conversation import Chat
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU





device = 'cuda:0'

def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
# parser.add_argument("--cfg-path", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--rc", type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--env", type=str)


args = parser.parse_args()
cfg = Config(args)
import os

path_benignImages = f'/home2/palash/p1_Jailbreak/LLaVA-NeXT/images/Memes/BENIGN_visual/'
path_toxicImages = f'/home2/palash/p1_Jailbreak/LLaVA-NeXT/images/Memes/TOXIC_visuals/'
path_memeImages = f'/home2/palash/p1_Jailbreak/LLaVA-NeXT/images/Memes/ISLAM_memes/'


all_benignImages = os.listdir(path_benignImages)
print(all_benignImages)

all_toxicImages = os.listdir(path_toxicImages)
print(all_toxicImages)

all_memeImages = os.listdir(path_memeImages)
print(all_memeImages)


query_list = []
with open(f'../jb_dataset/BeaverTails/BeaverTailsTestCases_100.txt', 'r') as f:
        queries = f.readlines()
        query_list = [i.replace('\n', '') for i in queries]

print(f'*** A total of {len(query_list)} queries.')

#############3

def get_modelAndProcessor():

    try:
        del model
        gc.collect()
        del processor
        gc.collect()
    except:
        pass

    model, vis_processor = init_model(args)
    model.eval()
    model = model.cuda().half()  # <-- Convert model to float16 (half precision)


    return model, vis_processor

def compute_average_cross_attention(attentions, image_token_mask):
    """
    Computes the average cross-attention given to image tokens by text tokens.
    
    Parameters:
    - attentions: Tensor of (num_layers, batch_size, num_heads, seq_len (query), seq_len (key))
    - image_token_mask: Tensor of shape (batch_size, seq_len), where True indicates image tokens.
    
    Returns:
    - avg_attention_per_layer: Tensor of shape (num_layers, num_heads), average attention to image tokens.
    """
    num_layers, batch_size, num_heads, seq_len, _ = attentions.shape
    # Shape: (32, 1, 32, 586, 586)

    # Remove batch dimension
    attentions = attentions.squeeze(1)  # Shape: (32, 32, 586, 586)

    # Identify image and text tokens
    image_token_indices = torch.where(image_token_mask.squeeze(0))[0]  # Indices of image tokens
    text_token_indices = torch.where(~image_token_mask.squeeze(0))[0]  # Indices of text tokens

    # Get attention from text tokens to image tokens: (num_layers, num_heads, num_text_tokens, num_image_tokens)
    text_to_image_attn = attentions[:, :, text_token_indices, :][:, :, :, image_token_indices]

    # Average across text tokens, image tokens and heads: Shape (num_layers)
    avg_attention_per_layer = text_to_image_attn.mean(dim=(1, 2, 3))
    
    return avg_attention_per_layer  # Shape: (num_layers)





def generate_output(prompt, image_path, model, processor, max_new_tokens):

    image = Image.open(image_path).convert('RGB')
    image_tensor = processor(image).unsqueeze(0).cuda().half()  # <-- Convert image to float16

    prompt = f"### <Img><ImageHere></Img>###Assistant:"

    texts = [prompt]

    with torch.no_grad():
        outputs = model.generate(
            image_tensor, texts, 
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_attentions=True,
        
        )

    attentions = [[att.detach().cpu() for att in attn] for attn in outputs['attentions']]
    # with torch.no_grad():  # Ensure no gradients are computed
    #     # op = model(**inputs)
    #     # print(op.keys())
    #     image_token_mask = (inputs["input_ids"] == 32000).int().detach().cpu()
    #     # image_token_mask = (model(**inputs).image_token_mask).detach().cpu()
        
    # return (attentions, image_token_mask)

model, processor = get_modelAndProcessor()


import random

benign_list = []
toxic_list = []
memes_list = []

for idx, ip in enumerate( query_list):
    
    prompt = f"<image>\nUSER: Answer the following question:\n{ip} ASSISTANT:" 
    
    path = path_benignImages + random.choice(all_benignImages)
    benign_images = path

    path = path_toxicImages + random.choice(all_toxicImages)
    toxic_images = path

    path = path_memeImages + random.choice(all_memeImages)
    meme_images = path

    benign_list.append((prompt, benign_images))
    toxic_list.append((prompt, toxic_images))
    memes_list.append((prompt, meme_images))

    break
    

benign_attns = []

max_new_tokens = 1
# Process benign inputs
i=0
for prompt, image_path in benign_list:
    i+=1
    print(i)
    attn , i_t_m = generate_output(prompt, image_path, model, processor, max_new_tokens)
    attn = torch.stack(attn[0])
    attn = compute_average_cross_attention(attn, i_t_m)
    benign_attns.append(attn)
    print(f"Output of benign prompt {i} generated")

torch.save(benign_attns,"benign_attention_weights.pt")
benign_attns = [] # Clear the weights


# CUDA_VISIBLE_DEVICES=0 python3 using_attentionForVisualization.py --env EL --cfg-path '/home2/palash/p3_metaphorExplanation/MiniGPT4/eval_configs/minigpt4_eval.yaml'
