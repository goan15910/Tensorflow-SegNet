from simple_segnet import Simple_SegNet
from vgg16_segnet import VGG16_SegNet
from vgg16_lr_segnet import VGG16_LR_SegNet
from vgg16_mr_segnet import VGG16_MR_SegNet
from vgg16_partial_segnet import VGG16_Partial_SegNet
from simple_lstm_segnet import Simple_LSTM_SegNet
from vgg16_lstm_segnet import VGG16_LSTM_SegNet
from vgg16_lstm_mr_segnet import VGG16_LSTM_MR_SegNet
from vgg16_lstm_lr_segnet import VGG16_LSTM_LR_SegNet
from vgg16_lstm_partial_segnet import VGG16_LSTM_Partial_SegNet
from vgg16_naive_fs_mr_segnet import VGG16_NAIVE_FS_MR_SegNet
from vgg16_fs_mr_segnet import VGG16_FS_MR_SegNet

nets_table = {
    'simple_segnet': Simple_SegNet, \
    'vgg16_segnet': VGG16_SegNet, \
    'vgg16_lr_segnet': VGG16_LR_SegNet, \
    'vgg16_mr_segnet': VGG16_MR_SegNet, \
    'vgg16_partial_segnet': VGG16_Partial_SegNet, \
    'simple_lstm_segnet': Simple_LSTM_SegNet, \
    'vgg16_lstm_segnet': VGG16_LSTM_SegNet, \
    'vgg16_lstm_mr_segnet': VGG16_LSTM_MR_SegNet, \
    'vgg16_lstm_lr_segnet': VGG16_LSTM_LR_SegNet, \
    'vgg16_lstm_partial_segnet': VGG16_LSTM_Partial_SegNet, \
    'vgg16_naive_fs_mr_segnet': VGG16_NAIVE_FS_MR_SegNet, \
    'vgg16_fs_mr_segnet': VGG16_FS_MR_SegNet, \
}
