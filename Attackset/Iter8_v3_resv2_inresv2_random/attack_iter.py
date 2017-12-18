"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
from functools import partial
from multiprocessing import Pool
import tensorflow as tf

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v4', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_resnet_v2', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'use_existing', 0, 'whether reuse existing result')

tf.flags.DEFINE_integer(
    'random_eps', 0, 'whether use random pertubation')

tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_string(
    'gpu','0','')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images

def save_images(arg):
    image,filename,output_dir = arg
    imsave(os.path.join(output_dir, filename), (image + 1.0) * 0.5, format='png')

def graph(x, y, i, x_max, x_min, grad, eps_inside):
  num_iter = FLAGS.num_iter
  alpha = eps_inside / num_iter
  momentum = FLAGS.momentum
  num_classes = 1001

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_v3, end_points_v3 = inception_v3.inception_v3(
        x, num_classes=num_classes, is_training=False)


  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
        x, num_classes=num_classes, is_training=False)

  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits_resnet, end_points_resnet = resnet_v2.resnet_v2_50(
        x, num_classes=num_classes, is_training=False)
            
  pred = tf.argmax(end_points_v3['Predictions'] + end_points_res_v2['Predictions'] + end_points_resnet['predictions'], 1)

  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)

  logits = (logits_v3 + logits_res_v2 +  logits_resnet) / 3
  auxlogits = (end_points_v3['AuxLogits'] + end_points_res_v2['AuxLogits'] ) / 2
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)
  cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                   auxlogits,
                                                   label_smoothing=0.0,
                                                   weights=0.4)
  noise = tf.gradients(cross_entropy, x)[0]
  x = x + alpha * tf.sign(noise)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, y, i, x_max, x_min, noise, eps_inside

def stop(x, y, i, x_max, x_min, grad, eps_inside):
  num_iter = FLAGS.num_iter
  return tf.less(i, num_iter)


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  print(FLAGS.output_dir)
  #eps = 2.0 * FLAGS.max_epsilon / 255.0
  gpus = np.array(FLAGS.gpu.split(',')).astype('int')
  n_gpus = len(gpus)
  bs_single = FLAGS.batch_size
  FLAGS.batch_size *= n_gpus
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  batch_shape_single = [bs_single, FLAGS.image_height, FLAGS.image_width, 3]
  tf.logging.set_verbosity(tf.logging.INFO)
  pool = Pool()
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    flists = set([f for f in os.listdir(FLAGS.input_dir) if 'png' in f])
    if FLAGS.use_existing == 1:
        flists_existing = set([f for f in os.listdir(FLAGS.output_dir) if 'png' in f ])
        newfiles = list(flists.difference(flists_existing))
        newfiles = [os.path.join(FLAGS.input_dir,f) for f in newfiles]
    else:
        newfiles = [os.path.join(FLAGS.input_dir,f) for f in flists]
    print('creating %s new files'%(len(newfiles)))
    if len(newfiles) == 0:
        return
    filename_queue = tf.train.string_input_producer(newfiles, shuffle = False, num_epochs = FLAGS.batch_size)
    image_reader = tf.WholeFileReader()
    filename, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_png(image_file)
    image.set_shape((299, 299, 3))

    eps = tf.placeholder(dtype='float32', shape = [FLAGS.batch_size, None, None, None])
    # Generate batch
    num_preprocess_threads = 20
    min_queue_examples = 256
    images,filenames = tf.train.batch(
        [image,filename],
        batch_size=FLAGS.batch_size,
        num_threads=num_preprocess_threads,
        capacity= 3 * FLAGS.batch_size,
        allow_smaller_final_batch = False)
    images = tf.cast(images,tf.float32)/255.0*2.-1.
    images_splits = tf.split(axis=0, num_or_size_splits=n_gpus, value=images)
    eps_splits = tf.split(axis=0, num_or_size_splits=n_gpus, value=eps)
 

    # Prepare graph
    #x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_advlist = []
    for i_gpu in range(n_gpus):
        start = i_gpu*bs_single
        print('gpu'+str(i_gpu))
        with tf.device('/gpu:'+str(i_gpu)):
          with tf.variable_scope(tf.get_variable_scope(),
                                 reuse=True if i_gpu > 0 else None):
    #      with tf.name_scope('%s_%d' % ('tower', i_gpu)):
            x_in_single = images_splits[i_gpu]
            eps_single = eps_splits[i_gpu]
            x_max = tf.clip_by_value(x_in_single + eps_single, -1.0, 1.0)
            x_min = tf.clip_by_value(x_in_single - eps_single, -1.0, 1.0)
            bs_this = x_in_single.shape[0]
            y = tf.constant(np.zeros([bs_single]), tf.int64)
            i = tf.constant(0)
            grad = tf.zeros_like(x_in_single)
            x_adv, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_in_single, y, i, x_max, x_min, grad, eps_single])
            x_advlist.append(x_adv)
    x_adv = tf.concat(x_advlist,0)
    # Run computation
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
    s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
    s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())  

    with tf.Session() as sess:
      sess.run(init)
      s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
      s6.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
      s8.restore(sess, FLAGS.checkpoint_path_resnet)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      n_iter = -(-(len(newfiles))//FLAGS.batch_size)
      stack_img = []
      stack_names = []
      for i in range(n_iter):
          if FLAGS.random_eps:
            eps_value = np.random.randint(1,FLAGS.max_epsilon, [FLAGS.batch_size,1,1,1])
          else:
            eps_value = np.ones([FLAGS.batch_size,1,1,1]) * FLAGS.max_epsilon
          eps_value = eps_value.astype('float32') *2 /255
          names,adv_images,orig_images = sess.run([filenames,x_adv,images], feed_dict={eps:eps_value})
          names = [os.path.basename(name) for name in names]
          stack_img.append(adv_images)
          stack_names.append(names)
          # save_images2(adv_images, names, FLAGS.output_dir, pool) 
          # save_images(adv_images, names, FLAGS.output_dir)
          if ((i+1)%100 ==0) or i == n_iter-1:
            print("%d / %d"%(i+1,n_iter)) 
            stack_img = np.concatenate(stack_img)
            stack_names = np.concatenate(stack_names)
            #partial_save = partial(save_one,images=stack_img,filenames=stack_names,output_dir=FLAGS.output_dir)
            paras = ((im,name,FLAGS.output_dir) for (im,name) in zip(stack_img,stack_names))
            pool.map_async(save_images,paras)
            stack_img = []
            stack_names = []


  #    save_images(adv_images, filenames, FLAGS.output_dir)
      # Finish off the filename queue coordinator.
      coord.request_stop()
      coord.join(threads)
    pool.close()
    pool.join()

if __name__ == '__main__':
  tf.app.run()
