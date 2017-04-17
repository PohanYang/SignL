import tensorflow as tf

sess = tf.Session()
new_saver = tf.train.import_meta_graph('Model/NIN-Model-Demo-3.meta')
#sess.run(tf.global_variables_initializer())
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
#all_vars = tf.get_collection()
