# loss function
import tensorflow as tf
import keras.backend as kb


class Total_loss:
    """
        tatal loss function (average between pixel loss and contextual loss)
        
        :param n_kernel: number of batches per epoch that use while training
        :type batches: int
    """
    def __init__(self, batches):
        self.batches = batches
        
    def pixel_loss(self, y_true, y_pred):
        """
            Calculate pixel loss
            
            :param y_true: y true
            
            :param y_pred: y predict
            
            :return: calculated pixel loss
        """
        return kb.mean(kb.abs(y_true - y_pred))


    def contextual_loss(self, y_true, y_pred):
        """
            Calculate contextual loss
            
            :param y_true: y true
            
            :param y_pred: y predict
            
            :return: calculated contextual loss
        """
        a = tf.image.rgb_to_grayscale(tf.slice(y_pred, [0, 0, 0, 0], [self.batches, 256, 256, 3]))
        b = tf.image.rgb_to_grayscale(tf.slice(y_true, [0, 0, 0, 0], [self.batches, 256, 256, 3]))

        y_pred = tf.divide(tf.add(tf.reshape(a, [tf.shape(a)[0], -1]), 1), 2)
        y_true = tf.divide(tf.add(tf.reshape(b, [tf.shape(b)[0], -1]), 1), 2)

        p_shape = tf.shape(y_true)
        q_shape = tf.shape(y_pred)

        p_ = tf.divide(y_true, tf.tile(tf.expand_dims(tf.reduce_sum(y_true, axis=1), 1), [1,p_shape[1]]))
        q_ = tf.divide(y_pred, tf.tile(tf.expand_dims(tf.reduce_sum(y_pred, axis=1), 1), [1,p_shape[1]]))
            
        return tf.reduce_sum(tf.multiply(p_, tf.math.log(tf.divide(p_, q_))), axis=1)


    def total_loss(self, y_true, y_pred):
        """
            Calculate total loss
            
            :param y_true: y true
            
            :param y_pred: y predict
            
            :return: calculated total loss
        """
        pix_loss = self.pixel_loss(y_true, y_pred)
        cont_loss = self.contextual_loss(y_true, y_pred)
        return (0.2 * pix_loss) + (0.8 * cont_loss)
    
    
    def get_pixel_loss_func(self):
        """
            Get pixel loss function
            
            :return: pixel loss function
            :rtype: method
        """
        return self.pix_loss
    
    
    def get_contextual_loss_func(self):
        """
            Get contextual function
            
            :return: contextual loss function
            :rtype: method
        """
        return self.contextual_loss
    
    
    def get_total_loss_func(self):
        """
            Get total function
            
            :return: toal loss function
            :rtype: method
        """
        return self.total_loss
    