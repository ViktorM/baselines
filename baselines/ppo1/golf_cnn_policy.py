from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

class golf_cnn_policy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, hid_size, filter_sizes, filter_nb, filter_strides, filter_activation,im_channels,im_dim, other_input_length):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, hid_size, filter_sizes, filter_nb, filter_strides, filter_activation,im_channels,im_dim, other_input_length)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, filter_sizes, filter_nb, filter_strides, filter_activation,im_channels,im_dim, other_input_length):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        #we need to get the observation vector, and slice it into the joint input and the image input
        #Input is of shape [None, other_input_length + im_dim[0]*im_dim[1]]
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + [ob_space.shape[0]]) 
        
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        other_data = tf.slice(ob,[0,0],[-1,other_input_length])
        flat_image = tf.slice(ob,[0,other_input_length],[-1,-1])
        
        #there is a potential bug here if we ever use an image with more than one channel
        #we MUST make sure we flatten it in the same way as it would be reshaped
        image = tf.reshape(flat_image, [tf.shape(ob)[0], im_dim[0], im_dim[1],1])
        
        #iterate over network properties, forming the conv layers
        last_feature = image
        prefix = "convLayer"
        suffix = 0
        for size, number, strides in zip(filter_sizes, filter_nb, filter_strides):
            name = prefix + str(suffix)
            last_feature = tf.nn.elu(U.conv2d(last_feature,number,name,[size, size], [strides, strides], pad="VALID"))
            suffix += 1
        
        last_feature = U.flattenallbut0(last_feature)
        
        last_layer = tf.concat([other_data, last_feature],axis=-1)
        prefix= "FClayer"
        for i in range(len(hid_size)-1):
            name = prefix + str(i)
            last_layer = tf.nn.elu(U.dense(last_layer, hid_size[i], name)) 
        
        
        last_layer = tf.nn.tanh(U.dense(last_layer, hid_size[-1],"last")) 
        
        v_layer = last_layer
        
        self.vpred = U.dense(v_layer, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]
		
        mean = U.dense(last_layer, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
        logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
        pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        
        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

