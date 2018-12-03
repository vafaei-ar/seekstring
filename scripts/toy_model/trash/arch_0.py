import tensorflow as tf

def architecture(x_in,drop_out):

    xp = tf.layers.conv2d(x_in,filters=8,kernel_size=5,strides=(1, 1),padding='same')
    xp = tf.layers.batch_normalization(xp)
    x1 = tf.nn.relu(xp)

    xp = tf.layers.conv2d(x1,filters=8,kernel_size=5,strides=(1, 1),padding='same')
    xp = tf.layers.batch_normalization(xp)
    xp = tf.nn.relu(xp)

#    xp = tf.layers.conv2d(xp,filters=8,kernel_size=5,strides=(1, 1),padding='same')
#    xp = tf.layers.batch_normalization(xp)
#    xp = tf.nn.relu(xp)

#    xp = xp+x1
#    xp = tf.layers.conv2d(xp,filters=8,kernel_size=5,strides=(1, 1),padding='same')
#    xp = tf.layers.batch_normalization(xp)
#    xp = tf.nn.relu(xp)

#    xp = tf.layers.dropout(xp, drop_out)
    x_out = tf.layers.conv2d(xp,filters=1,kernel_size=5,strides=(1, 1),padding='same')

    return x_out
