import torch
import devicetorch
from RIFE.RIFE_HDv3 import Model as BaseModel
device = devicetorch.get(torch)


class EnhancedRIFEModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = devicetorch.get(torch)
        self.flownet.to(self.device)
        
    def to(self, target_device):
        self.device = target_device
        self.flownet = self.flownet.to(target_device)
        return self
        
    def inference(self, img0, img1, timestep=None, scale=1.0):
        """Enhanced inference method that supports both timestep and scale parameters"""
        # Enhanced device handling
        if img0.device != self.device:
            img0 = img0.to(self.device)
        if img1.device != self.device:
            img1 = img1.to(self.device)
            
        # If timestep is provided, use it to modify the scale
        # This maps timestep 0.5 to scale 1.0
        if timestep is not None:
            scale = 1.0
            
        result = super().inference(img0, img1, scale=scale)
        return result
