from models.models import *
from utils.datasets import *
from utils.general import *


cfg = "/home/dong9/PycharmProjects/CottonWeed_Detection/PyTorch_YOLOv4-master/cfg/yolov4-csp-x-leaky.cfg"
weights = "yolov4-csp-x-leaky.weights"
model = Darknet(cfg)

if weights.endswith('.pt'):
   model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
   save_weights(model, path='converted.weights', cutoff=-1)
   print("Success: converted '%s' to 'converted.weights'" % weights)

elif weights.endswith('.weights'):
   _ = load_darknet_weights(model, weights)
   chkpt = {'epoch': -1, 'best_loss': None, 'model': model.state_dict(), 'optimizer': None}
   torch.save(chkpt, 'yolov4-csp-x-leaky.pt')
   print("Success: converted '%s' to 'yolov4-csp-x-leaky.pt'" % weights)
