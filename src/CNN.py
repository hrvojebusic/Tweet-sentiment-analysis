import tensorflow as tf

class CNN(object):
    """
    Class for creating a tensorflow graph for classification using convolutional neural networks
    """
    def __init__(self):
        self.n_filters = 128                 
        self.filter_sizes = [2, 3, 4, 5, 6]
        self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
              
        n_filters_total = self.n_filters*len(self.filter_sizes)
        
        self.x = tf.placeholder(tf.float32, [None, 40, 200,  1], name='embedding') 
        self.y = tf.placeholder(tf.float32, [None, 2], name='class_probability')        	
        
        layers = []
        for i, size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % size):
                Wname = 'W_conv_%s' % size
                W = tf.get_variable(Wname, shape=[size, 200, 1 ,self.n_filters], initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.2, shape=[self.n_filters]), name='b')     
                pre_act = tf.nn.bias_add(tf.nn.conv2d(self.x, W, strides=[1,1,1,1],padding='VALID',name='conv'), b, name='pre_activation')                                        
                act = tf.nn.relu(pre_act, name='activation')
                pooled = tf.nn.max_pool(act, ksize=[1,40-size+1, 1, 1], strides=[1,1,1,1], padding='VALID', name='max_pooled')  
                layers.append(pooled)
                
        self.features = tf.reshape(tf.concat(layers, 3), [-1, n_filters_total], name='all_features')
                                        
        with tf.name_scope('dropout'):
            self.features_dropout = tf.nn.dropout(self.features, self.dropout_prob, name='after_dropout_1')
            softmax_input = self.features_dropout
            n_in_softmax = n_filters_total
            print(n_filters_total)

        with tf.name_scope('softmax'):
            W = tf.get_variable('W_softmax', shape=[n_in_softmax, 2], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name='b_softmax')
            self.scores = tf.nn.softmax(tf.nn.xw_plus_b(softmax_input, W, b), name='prediction_probabilites')
            self.y_pred = tf.argmax(self.scores,  axis=1,  name = 'predicted_classes')
        
        with tf.name_scope('loss_calculation'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y), name='loss')
            
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.y_pred, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')