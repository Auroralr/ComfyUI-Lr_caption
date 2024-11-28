from .Lr_caption_node import Lr_caption
from .Lr_caption_node import Lr_caption_load

NODE_CLASS_MAPPINGS = {
    "Lr_Lr_caption_load":Lr_caption_load,
    "Lr_caption":Lr_caption
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lr_Lr_caption_load":"🐾Lr_caption_load",
    "Lr_caption":"🐾Lr_caption"
}