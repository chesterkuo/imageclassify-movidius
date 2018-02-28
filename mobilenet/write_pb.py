from __future__ import print_function
import tensorflow as tf
from mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
from tensorflow.python.framework import graph_util
slim = tf.contrib.slim

def run():
    image_size = 224
    num_classes = 200
    logdir = './log'

    checkpoint_file = tf.train.latest_checkpoint(logdir)

    with tf.Graph().as_default() as graph:
        
        images = tf.placeholder("float", [1, image_size, image_size, 3], name="input")
        #images = tf.placeholder(shape=[1, image_size, image_size, 3], dtype=tf.float32, name = 'Placeholder_only')

        with slim.arg_scope(mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1(images, num_classes = num_classes, is_training = False)

        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        #Setup graph def
        input_graph_def = graph.as_graph_def()
        output_node_names = "MobilenetV1/Predictions/Softmax"
        output_graph_name = "./frozen_model_mobilenet_v1.pb"

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

run()
