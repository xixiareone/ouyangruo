import numpy as np
import tensorflow as tf
import cPickle as pickle
from net3 import graph, get_args,model_path


args = get_args()
with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
    ckpt = tf.train.get_checkpoint_state(model_path)
    print(ckpt)
    assert (ckpt and ckpt.model_checkpoint_path)
    saver.restore(session, ckpt.model_checkpoint_path)
    pickle.dump(tuple(v.eval() for v in args), open('/home/ouyangruo/Documents/BiShe/Model_one/args.pickle', 'wb'), 2)
        