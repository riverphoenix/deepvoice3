import tensorflow as tf
import argparse
from data_load import get_batch
from hyperparams import Hyperparams as hp
from utils import *
import numpy as np
from time import gmtime, strftime

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="logs/export_test/frozen_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    g = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    # for op in graph.get_operations():
    #     print(op.name)
        
    # We access the input and output nodes 
    x = g.get_tensor_by_name('prefix/mel_inp_pl:0')
    y = g.get_tensor_by_name('prefix/converter/mag_out:0')

    # args.data_paths = hp.data
    x_test = np.random.random((16,810,80))
    print(x_test.shape)
        
    # We launch a Session
    with tf.Session(graph=g) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 
        #mag, y = sess.run([g.mag_output, g.y],feed_dict={ g.mel: x_test })
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        y_out = sess.run(y, feed_dict={ x: x_test })
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        print(y_out)

        #loss = tf.reduce_mean(tf.abs(y_out - y_real))
        #print(loss)

        # plot_loss(args,y_out,y_real,0)
        # wav_mag, wav_y = get_wavs(args,y_out,y_real,0)
        # plot_wavs(args,wav_mag,wav_y,0)