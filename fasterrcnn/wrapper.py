from fasterrcnn.utils.config import cfg

class FasterRCNNWrapper():
    def __init__(self, device, model):
        self.device = device
        self.model = model
    