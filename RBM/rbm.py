import tensorflow as tf
import numpy as np
import logging




class Placeholders(object):
    def __init__(self):
        pass

class RBM(object):
    def __init__(self, visible_size, hidden_size):
        self._sess = None
        self.__counters = []
        self.__restore = False
        self._ph = Placeholders()
        self._ph.visible_in = tf.placeholder(tf.float32, shape=[32, visible_size], name='visible_in')
        self._ph.hidden_in = tf.placeholder(tf.float32, shape=[32, hidden_size], name='hidden_in')


        with tf.variable_scope("parameters"):
            self.W = tf.get_variable("weights",
                                     shape=[visible_size, hidden_size],
                                     trainable=True,
                                     initializer=tf.random_normal_initializer(
                                        mean=0.0,
                                        stddev=0.1,
                                        )
                                     )
            self.b_h = tf.get_variable("hidden_biases",
                                       shape=[hidden_size,],
                                       trainable=True,
                                       initializer=tf.zeros_initializer(),
                                       )

            self.b_v = tf.get_variable("visible_biases",
                                       shape=[visible_size,],
                                       trainable=True,
                                       initializer=tf.zeros_initializer(),
                                      )

        self._optimize_op = self._get_optimize_op(v_init=self._ph.visible_in)


        _, hs = self._sample_h(v=self._ph.visible_in)
        _, vs = self._sample_v(h=hs)
        self._inference_op = vs

    def train(self, v_init):
        self.sess.run(self._optimize_op, feed_dict={self._ph.visible_in:v_init})



    def inference(self, v_init):
        return self.sess.run(self._inference_op, feed_dict={self._ph.visible_in: v_init})

    def _get_optimize_op(self, v_init):
        """v_init - visible nodes"""
        v0, vk, ph0, phk = self._contrastive_divergence_gibbs_sampling(v_init, k=5)
        dW, db_h, db_v = self._compute_gradients(v0, vk, ph0, phk)

        #self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        #grads_and_vars = [ [dW, self.W], [db_h, self.b_h], [db_v, self.b_v] ]
        #optimize_op = self._optimizer.apply_gradients(grads_and_vars)
        lr = 1e-2
        optimize_op = [ tf.assign(self.W, self.W + lr*dW),
                        tf.assign(self.b_h, self.b_h + lr*db_h),
                        tf.assign(self.b_v, self.b_v + lr*db_v)
                      ]
        return optimize_op

    def _bernoulli_sample(self, p):
        """Using an input vector p of probabilities, return a same-sized vector
           of binary values sampled from the probability distribution p"""
        return tf.where(tf.less(p, tf.random_uniform(shape=p.shape)), x=tf.zeros_like(p), y=tf.ones_like(p))

    def _sample_h(self, v):
        """Given a binary input vector v, return the state of h, e.g. the
           probabilities that a hidden neuron is activated
           Returns:
             prob_h: probability of each hidden unit being activated
             samp_h: an h_vector sampled from prob_h
           """
        prob_h = tf.sigmoid(tf.nn.bias_add(tf.matmul(v, self.W), self.b_h))
        samp_h = self._bernoulli_sample(prob_h)
        return prob_h, samp_h

    def _sample_v(self, h):
        """Given a hidden vector, return the probability vector v,
           and sample v and return a sample
        """
        W_T = tf.transpose(self.W, [1,0]) #tranpose of W

        prob_v = tf.sigmoid(tf.nn.bias_add(tf.matmul(h, W_T), self.b_v))
        samp_v = self._bernoulli_sample(prob_v)
        return prob_v, samp_v

    def _contrastive_divergence_gibbs_sampling(self, v_init, k):
        """
        Procedure outline:

            set v_n = v_init
            for n in 0..k:
                set h_n = 1 with probability sigmoid(W' * v_n + b_h )
                set v_n = 1 with probability sigmoid(W  * h_n + b_v )
        """

        v_ = v_init
        for n in range(k):
            p_h_k, h_ = self._sample_h(v_)  #returns (probabilies, sample)
            if n==0:
                p_h_init = p_h_k
            _, v_ = self._sample_v(h_)

        """We need some specific information for gradient updates, so let's assemble that:"""
        v_init = v_init
        p_h_init = p_h_init
        v_k = v_
        p_h_k, _ = self._sample_h(v_k)

        return v_init, v_k, p_h_init, p_h_k



    def _compute_gradients(self, v_init, v_k, p_h_init, p_h_k):
        """
        v_init:     visible units before gibbs sampling
        v_k:        visible units after k iterations of gibbs sampling
        p_h_init:   probabilies of the hidden layers before gibbs sampling
        p_h_k:      probabilities of the hideen layers after gibbs sampling
        """

        #do the outer products, sum along the batch dimension
        dW = tf.einsum('bi,bj->ij', v_init, p_h_init) - tf.einsum('bi,bj->ij', v_k, p_h_k)

        dW = dW / tf.cast(p_h_k.shape[0], tf.float32) #divide by batch size:

        db_h = tf.reduce_mean(p_h_init - p_h_k, axis=0)
        db_v = tf.reduce_mean(v_init - v_k, axis=0)

        return dW, db_h, db_v

    @property
    def sess(self):
        if self._sess is None:
            logging.error("You must attach a session using the attach_session(sess) method.")
            sys.exit(1)
        else:
            return self._sess

    def attach_session(self, sess):
        self._sess = sess
        for counter in self.__counters:
            counter.attach_session(self._sess)

        if self.__restore:
            try:
            #if True:
                logging.debug("Attempting to restore variables")
                latest_checkpoint = tf.train.latest_checkpoint(self._checkpoint_path)
                self._saver.restore(sess, latest_checkpoint)
                logging.info("Variables restored from checkpoint {}".format(latest_checkpoint))
            except:
                logging.warning("Variable restoration/ FAILED. Initializing variables instead.")
                sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())

#        self._summary_writer = tf.summary.FileWriter(get_log_path('./logs',self._flags.get('name_prefix', 'run_')),
#                                                     self._sess.graph, flush_secs=1)
        self._merged_summaries = tf.summary.merge_all()




class EnergyBasedModel(object):
    def __init__(self, flags):
        self._flags = flags
