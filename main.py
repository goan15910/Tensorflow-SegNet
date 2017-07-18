import tensorflow as tf
from graph_runner import Graph_Runner

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('testing', '', """ checkpoint file """)
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file path """)
tf.app.flags.DEFINE_string('pretrained', None, """ pretrained weight npy file path """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)
tf.app.flags.DEFINE_string('net', "simple_segnet", """ used net name """)
tf.app.flags.DEFINE_string('dataset', "camvid", """ used dataset """)
tf.app.flags.DEFINE_string('log_dir', "/tmp3/jeff/TensorFlow/SegNet/", """ dir to store ckpt """)
tf.app.flags.DEFINE_string('image_dir', "/tmp3/jeff/CamVid/train.txt", """ path to CamVid image """)
tf.app.flags.DEFINE_string('test_dir', "/tmp3/jeff/CamVid/test.txt", """ path to CamVid test image """)
tf.app.flags.DEFINE_string('val_dir', "/tmp3/jeff/CamVid/val.txt", """ path to CamVid val image """)
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
    print("Initial lr: %f"%FLAGS.learning_rate)
    print("{} Image dir: {}".format(FLAGS.dataset, FLAGS.image_dir))
    print("{} Val dir: {}".format(FLAGS.dataset, FLAGS.val_dir))

  print("Log dir: %s"%FLAGS.log_dir)


def main(args):
    checkArgs()
    g_runner = Graph_Runner(FLAGS)
    if FLAGS.testing:
      g_runner.testing()
    else:
      g_runner.training()

if __name__ == '__main__':
  tf.app.run()
