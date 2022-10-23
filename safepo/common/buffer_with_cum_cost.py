import numpy as np
import torch

class Buffer_With_Cum_Cost:
    def __init__(self, buffer):
        self.buffer = buffer

    def __getattr__(self, func):
        return getattr(self.buffer, func)
    
    def get(self):
        data = self.buffer.get()
        cost_buf = np.cumsum(self.buffer.cost_buf[::-1], axis=0)[::-1].copy()
        data["cum_cost"] = torch.as_tensor(cost_buf, dtype=torch.float32)

        return data
