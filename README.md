# Tensorflow-SegNet
Implement slightly different (see below for detail) [SegNet](http://arxiv.org/abs/1511.00561) in tensorflow,
successfully trained segnet-basic in CamVid dataset.

Due to indice unravel still unavailable in tensorflow, the original upsampling
method is temporarily replaced simply by deconv( or conv-transpose) layer (without pooling indices).
You can follow the issue here: https://github.com/tensorflow/tensorflow/issues/2169
(The current workaround for unpooling layer is a bit slow because it lacks of GPU support.)

for model detail, please go to https://github.com/alexgkendall/caffe-segnet

# Requirement
tensorflow r1.2 <br />
Pillow (optional, for write label image) <br />
scikit-image <br />
easydict <br />
ImageIO <br />
numpy <br />
opencv <br />

# Usage
training:

  python main.py --mode train [--pretrained pretrained_npy_path] --net your_net_name --log_dir output_dir

finetune:

  python main.py --mode train --net your_net_name --finetune path_to_saved_ckpt --log_dir output_dir

testing:

  python main.py --mode test --net your_net_name --testing path_to_saved_ckpt --log_dir output_dir --save_image True


# Dataset
This Implement default to use CamVid dataset as described in the original SegNet paper,
The dataset can be download from author's github https://github.com/alexgkendall/SegNet-Tutorial in the CamVid folder

example format:

"path_to_image1" "path_to_corresponded_label_image1",

"path_to_image2" "path_to_corresponded_label_image2",

"path_to_image3" "path_to_corresponded_label_image3",

.......
