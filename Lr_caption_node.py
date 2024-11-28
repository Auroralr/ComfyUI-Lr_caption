import os

from transformers import AutoProcessor, MllamaForConditionalGeneration
import torch
from PIL import Image
import numpy as np
import re

import folder_paths # Comfyui 的源生库支持获取文件路径 # /ComfyUI/models
from model_management import get_torch_device
DEVICE = get_torch_device()

class LrPipeline:
    def __init__(self):
        self.vllm_model = None
        self.vllm_processor = None
        self.parent = None
    
    def clearCache(self):
        self.vllm_model = None
        self.vllm_processor = None

class Lr_caption_load:

    def __init__(self):
        self.model = None
        self.pipeline = LrPipeline()
        self.pipeline.parent = self
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", "meta-llama/Llama-3.2-11B-Vision-Instruct"],), 
            }
        }

    CATEGORY = "LRZ/VLLM"
    RETURN_TYPES = ("LrPipeline",)
    FUNCTION = "gen"
    
    def download_hg_model(self,model_id:str,exDir:str=''): # lib包 xmodel.py
    # 下载本地 # folder_paths.models_dir == /ComfyUI/models
        model_checkpoint = os.path.join(folder_paths.models_dir, exDir, os.path.basename(model_id))
        print("Checking model file path:",model_checkpoint)
        if not os.path.exists(model_checkpoint):
            print("Check that the model file does not exist and is being redownloaded")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
        return model_checkpoint

    def loadCheckPoint(self):
        # 清除一波
        if self.pipeline != None:
            self.pipeline.clearCache()

        # 多模态模型
        VLLM_PATH = self.download_hg_model(self.model,"VLLM") # 这个路径指向/ComfyUI/models/VLLM/目录下的Meta-Llama-3.1-8B-bnb-4bit文件夹
        vllm_model = MllamaForConditionalGeneration.from_pretrained(VLLM_PATH, device_map="auto", torch_dtype=torch.bfloat16)
        vllm_processor = AutoProcessor.from_pretrained(VLLM_PATH)
        vllm_model.eval()

        self.pipeline.vllm_model = vllm_model
        self.pipeline.vllm_processor = vllm_processor

    def clearCache(self):
        if self.pipeline != None:
            self.pipeline.clearCache()

    def gen(self,model):
        if self.model == None or self.model != model or self.pipeline == None:
            self.model = model
            self.loadCheckPoint()
        return (self.pipeline,)

class Lr_caption:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Lr_pipeline": ("LrPipeline",),
                "image": ("IMAGE",),
                "caption_method": (['caption', 'short_prompt', 'long_prompt'], {
                    "default": "caption"
                }),
                "prompt":   ("STRING", {"multiline": True, "default": "A descriptive caption for this image"},),
                "max_new_tokens":("INT", {"default": 1024, "min": 10, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cache": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "LRZ/VLLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"

    def update_prompt_default(self, caption_method):
        # 根据caption_method更新prompt的default值

        if caption_method == 'short_prompt':
            new_default = "Provide a detailed description of the details and content contained in the image, and generate a short prompt that can be used for image generation tasks in Stable Diffusion,you should only return prompt,itself without any additional information."
        elif caption_method == 'long_prompt':
            new_default = "Follow these steps to create a Midjourney-style long prompt for generating high-quality images: \
                1. The prompt should include rich details, vivid scenes, and composition information, capturing the important elements that make up the scene. \
                2. You can appropriately add some details to enhance the vividness and richness of the content, while ensuring that the long prompt does not exceed 256 tokens,you should only return prompt,itself without any additional information"
        else:
            new_default = None

        return new_default 
        
        # else:
            # new_default = "Describe this image in detail, focusing on the main elements, colors, and overall composition. After the description, generate a list of relevant tags that could be used for image generation task with Stable Diffusion."

    def tensor2pil(self, t_image: torch.Tensor) -> Image:  # lib 包
        # .squeeze() 方法从 NumPy 数组的形状中移除所有长度为 1 的维度。这通常用于去除批次大小维度，当处理单个图像时这个维度是不必要的。
        # np.clip(..., 0, 255)：np.clip() 函数确保 NumPy 数组中的所有值都在指定的范围内，这里是 [0, 255]。如果数组中的任何值小于 0 或大于 255，它们将被分别设置为 0 或 255。
        # .astype(np.uint8) 将 NumPy 数组的数据类型转换为 np.uint8，这是表示图像像素值的常用数据类型，因为它可以表示 0 到 255 之间的整数值。
        return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
    def gen(self,Lr_pipeline,image,caption_method,prompt,max_new_tokens,temperature,cache):  # joy_pipeline
        
        if Lr_pipeline.vllm_processor == None :
            Lr_pipeline.parent.loadCheckPoint()

        caption_new_default = self.update_prompt_default(caption_method)
        
        if caption_new_default == None:
            caption_new_default = prompt

        messages = [
            [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": caption_new_default }
                    ]
                }
            ],
        ]
        
        current_dir = os.getcwd()
        print('1.image type:', image, type(image))
        print("2.Current directory:", current_dir)
        input_image = self.tensor2pil(image)
        print("3.image_tensor2pil type:",type(input_image))
        
        processor = Lr_pipeline.vllm_processor
        model = Lr_pipeline.vllm_model

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        # assert isinstance(text, str) # LLM
        inputs = processor(text=text, images=input_image, return_tensors="pt")
        inputs = inputs.to(DEVICE)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_k=10, temperature=temperature, suppress_tokens=None)
        str = processor.decode(output[0], skip_special_tokens=True)[len("\nuser\n"+caption_new_default+".assistant\n"):] # 7 代表 /n + user + /n ... + /n
        print("Ask a question:\n", caption_new_default)
        str = re.sub(r'[*+]', '', str)
        str = re.sub(r'\.\s*', '. ', str)
        str = re.sub(r'\:\s*', ': ', str)
        print("answer:\n", str)

        # # input_image.show() # shell没有图像查看器 fromarray 两个参数，一个array另一个，model="通道" “1”, “L”, “RGB”, “RGBA”, “CMYK”, “YCbCr”, “LAB”, “HSV”, “I”, “F”
        # clip_processor = AutoProcessor.from_pretrained("./siglip-so400m-patch14-384")
        # # clip_processor = AutoProcessor.from_pretrained(r"C:\workspace\vspj\cfjiedian\siglip-so400m-patch14-384")
        # pImge = clip_processor(images=input_image, return_tensors='pt').pixel_values
        # # .pixel_values 是 AutoProcessor 类的一个属性，它返回经过预处理的图像像素值张量，这些张量已经准备好被模型使用。
        # pImge = pImge.to(torch.device('cpu'))
        
        if cache == False:
           Lr_pipeline.parent.clearCache() 

        return (str,)