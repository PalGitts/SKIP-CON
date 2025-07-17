from PIL import Image
import requests, transformers
from transformers import AutoProcessor, LlavaNextForConditionalGeneration 
import gc

from transformers.models.mistral.modeling_mistral import global_stackedHS
print(global_stackedHS)