from __future__ import print_function
import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from tensorflow.python.framework import graph_util
slim = tf.contrib.slim

def run():
    image_size = 299
    num_classes = 200
    logdir = './log'

    checkpoint_file = tf.train.latest_checkpoint(logdir)

    with tf.Graph().as_default() as graph:
        
        images = tf.placeholder(shape=[None, image_size, image_size, 3], dtype=tf.float32, name = 'Placeholder_only')

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images, num_classes = num_classes, is_training = False)

        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        #Setup graph def
        input_graph_def = graph.as_graph_def()
        output_node_names = "InceptionResnetV2/Predictions/Softmax"
        output_graph_name = "./frozen_model_inception_resnet_v2.pb"

        with tf.Session() as sess:
            saver.restore(sess, checkpoint_file)

            #Exporting the graph
            print ("Exporting graph...")
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(","))

            with tf.gfile.GFile(output_graph_name, "wb") as f:
                f.write(output_graph_def.SerializeToString())

if __name__=='_main__':
    run()
