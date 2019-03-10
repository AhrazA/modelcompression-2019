import torch.nn as nn

class MaskedSequential(nn.Sequential):
    # Applies masks per residual block
    def set_mask(self, masks):
        iter_masks = iter(masks)
        for index, layer in enumerate(self.children()):
            if 'Masked' in str(type(layer)):
                layer.set_mask(next(iter_masks))
