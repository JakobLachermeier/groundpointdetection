from torch.nn import Module
from dataloader import Instance, FrameSize
import numpy as np
import torch


class ScaleToImageSize(Module):
    def __init__(self, image_size):
        super(ScaleToImageSize, self).__init__()
        self.image_size = image_size

    def forward(self, x: Instance, frame_size: FrameSize):
        height_width = np.array([frame_size.height, frame_size.width]).reshape(1, 2)
        width_height = np.array([frame_size.width, frame_size.height])
        x.hull_pv /=  height_width
        x.gcp_pv /= width_height
        return x
    
class ToTensor(Module):
    def __init__(self):
        super(ToTensor, self).__init__()

    def forward(self, x: Instance):
        hull = torch.from_numpy(x.hull_pv)
        gcp = torch.from_numpy(x.gcp_pv)
        return hull, gcp
    
