from simple_segnet import Simple_SegNet
from vgg16_segnet import VGG16_SegNet
from vgg16_lr_segnet import VGG16_LR_SegNet
from vgg16_mr_segnet import VGG16_MR_SegNet
from vgg16_partial_segnet import VGG16_Partial_SegNet

nets_table = {
    'simple_segnet': Simple_SegNet, \
    'vgg16_segnet': VGG16_SegNet, \
    'vgg16_lr_segnet': VGG16_LR_SegNet, \
    'vgg16_mr_segnet': VGG16_MR_SegNet, \
    'vgg16_partial_segnet': VGG16_Partial_SegNet, \
}
