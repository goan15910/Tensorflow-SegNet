import tensorflow as tf
from graph_runner import Graph_Runner

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', None, """ Either train / test / extract """)
tf.app.flags.DEFINE_string('test_ckpt', None, """ checkpoint file path for testing """)
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file path """)
tf.app.flags.DEFINE_string('pretrained', None, """ pretrained weight npy file path """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)
tf.app.flags.DEFINE_string('net', "simple_segnet", """ used net name """)
tf.app.flags.DEFINE_string('dataset', "camvid", """ used dataset """)
tf.app.flags.DEFINE_string('log_dir', "/tmp3/jeff/TensorFlow/Simple_SegNet/", """ dir to store ckpt """)
tf.app.flags.DEFINE_string('image_dir', "/tmp3/jeff/CamVid/train.txt", """ path to CamVid image """)
tf.app.flags.DEFINE_string('test_dir', "/tmp3/jeff/CamVid/test.txt", """ path to CamVid test image """)
tf.app.flags.DEFINE_string('val_dir', "/tmp3/jeff/CamVid/val.txt", """ path to CamVid val image """)
tf.app.flags.DEFINE_boolean('save_predict', True, """whether to save prediction""")

def checkArgs():
  assert FLAGS.mode in ['train', 'test', 'extract'], \
      "Selected mode {} not supported".format(FLAGS.mode)
  if FLAGS.mode != 'test':
    print('The model is set to Testing')
    print("check point file: %s"%FLAGS.test_ckpt)
    print("{} testing dir: {}".format(FLAGS.dataset, FLAGS.test_dir))
  elif FLAGS.finetune != '':
    print('The model is set to Finetune from ckpt')
    print("check point file: %s"%FLAGS.finetune)
    print("{} Image dir: {}".format(FLAGS.dataset, FLAGS.image_dir))
    print("{} Val dir: {}".format(FLAGS.dataset, FLAGS.val_dir))
  elif FLAGS.mode == 'train':
    print('The model is set to Training')
    print("Initial lr: %f"%FLAGS.learning_rate)
    print("{} Image dir: {}".format(FLAGS.dataset, FLAGS.image_dir))
    print("{} Val dir: {}".format(FLAGS.dataset, FLAGS.val_dir))
  elif FLAGS.mode == 'extract':
    print('The model is set to extract features')
    print("{} extracting dir: {}".format(FLAGS.dataset, FLAGS.test_dir))

  print("Log dir: %s"%FLAGS.log_dir)


def main(args):
    checkArgs()
    g_runner = Graph_Runner(FLAGS)
    g_runner.run(FLAGS.mode)

if __name__ == '__main__':
  tf.app.run()
