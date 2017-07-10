import tensorflow as tf
from traintest import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('testing', '', """ checkpoint file """)
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file path """)
tf.app.flags.DEFINE_string('pretrained', '', """ pretrained weight npy file path """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)
tf.app.flags.DEFINE_string('net', "simple_segnet", """ used net name """)
tf.app.flags.DEFINE_string('dataset', "camvid", """ used dataset """)
tf.app.flags.DEFINE_string('log_dir', "/tmp3/jeff/TensorFlow/SegNet/", """ dir to store ckpt """)
tf.app.flags.DEFINE_string('image_dir', "/tmp3/jeff/CamVid/train.txt", """ path to CamVid image """)
tf.app.flags.DEFINE_string('test_dir', "/tmp3/jeff/CamVid/test.txt", """ path to CamVid test image """)
tf.app.flags.DEFINE_string('val_dir', "/tmp3/jeff/CamVid/val.txt", """ path to CamVid val image """)
tf.app.flags.DEFINE_integer('num_class', "11", """ total class number """)
tf.app.flags.DEFINE_boolean('load_seq', False, """ whether to load data by sequence """)
tf.app.flags.DEFINE_boolean('save_image', True, """ whether to save predicted image """)

def checkArgs():
    if FLAGS.testing != '':
        print('The model is set to Testing')
        print("check point file: %s"%FLAGS.testing)
        print("{} testing dir: {}".format(FLAGS.dataset, FLAGS.test_dir))
    elif FLAGS.finetune != '':
        print('The model is set to Finetune from ckpt')
        print("check point file: %s"%FLAGS.finetune)
        print("{} Image dir: {}".format(FLAGS.dataset, FLAGS.image_dir))
        print("{} Val dir: {}".format(FLAGS.dataset, FLAGS.val_dir))
    else:
        print('The model is set to Training')
        print("Max training Iteration: %d"%FLAGS.max_steps)
        print("Initial lr: %f"%FLAGS.learning_rate)
        print("{} Image dir: {}".format(FLAGS.dataset, FLAGS.image_dir))
        print("{} Val dir: {}".format(FLAGS.dataset, FLAGS.val_dir))

    print("Batch Size: %d"%FLAGS.batch_size)
    print("Log dir: %s"%FLAGS.log_dir)


def main(args):
    checkArgs()
    if FLAGS.testing:
        test(FLAGS)
    elif FLAGS.finetune:
        training(FLAGS, is_finetune=True)
    else:
        training(FLAGS, is_finetune=False)

if __name__ == '__main__':
  tf.app.run()
