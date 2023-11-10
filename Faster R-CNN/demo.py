import torch

from data.util import read_image
from model import FasterRCNNVGG16
from utils import array_tool
from utils.vis_tool import vis_bbox

PARAM_PTH = '/content/gdrive/MyDrive/Fudan/Copy of fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth'


img = read_image('/content/gdrive/MyDrive/Fudan/CV-object-detection/Faster R-CNN/imgs/demo.jpg')
img = torch.from_numpy(img)[None]
faster_rcnn = FasterRCNNVGG16()

state_dict = torch.load(PARAM_PTH)
faster_rcnn.load_state_dict(state_dict['model'])
_bboxes, _labels, _scores = faster_rcnn.predict(img, visualize=True)
vis_bbox(array_tool.tonumpy(img[0]), array_tool.tonumpy(_bboxes[0]), array_tool.tonumpy(_labels[0]).reshape(-1), array_tool.tonumpy(_scores[0]).reshape(-1))
# it failed to find the dog, but if you set threshold from 0.7 to 0.6, you'll find it