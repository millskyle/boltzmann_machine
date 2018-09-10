import tensorflow as tf
import numpy as np

class RBM(object):
    def __init__(self, FLAGS):
        self.h_size = FLAGS.h_size
        self.v_size = FLAGS.v_size
        self.batch_size = FLAGS.batch_size
        self.learning_rate = FLAGS.learning_rate
        with tf.variable_scope('RBM'):
            self.W = tf.get_variable('W', shape=(v_size, h_size), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            self.bh = tf.get_variable('b_h', shape=(h_size,), initializer=tf.zeros_initializer())
            self.bv = tf.get_variable('b_v', shape=(v_size,), initializer=tf.zeros_initializer())

    def _bernoulli_sample(self, p):
        """Using an input vector p of probabilities, return a same-sized vector
           of binary values sampled from the probability distribution p"""
        return tf.where(tf.less(p, tf.random.uniform(shape=p.shape)), x=tf.zeros_like(p), y=tf.ones_like(p))


    def _sample_h(self, v):
        """Given a binary input vector v, return the state of h, e.g. the
           probabilities that a hidden neuron is activated


           Returns:
             prob_h: probability of each hidden unit being activated
             samp_h: an h_vector sampled from prob_h
           """
        prob_h = tf.sigmoid(tf.nn.bias_add(tf.matmul(v, self.W), self.bh))
        samp_h = self._bernoulli_sample(prob_h)
        return prob_h, samp_h

    def _sample_v(self, h):
        """Given a hidden vector, return the probability vector v,
           and sample v and return a sample
        """

        prob_v = tf.sigmoid(tf.nn.bias_add(tf.matmul(h, tf.transpose(self.W, [1,0])), self.bv))
        samp_v = self._bernoulli_sample(prob_v)
        return prob_v, samp_v


    def _gibbs_sample(self, v):
        """do Gibbs sampling

        """

        def break_condition(i, vk, hk, v):
            return tf.less(i,k)[0]

        def loop_body(i, vk, hk, v):
            _, hk = self._sample_h(vk)
            _, vk = self.sample_v(hk)
            #make sure that any vk=-1 stay as -1
            vk = tf.where(tf.less(v,0),v,vk)
            return [i+1, vk, hk, v]

        #get initial h probabilities:
        ph0, _ = self._sample_h(v)

        vk = v
        hk = tf.zeros_like(ph0)

        i=0
        k=tf.constant([100])

        [i, vk, hk, v] = tf.while_loop(break_condition, loop_body, [i, vk, hk, v])

        #get the final h probabilities
        phk, _ = self.sample_h(vk)

        return v, vk, ph0, phk, i


    def _compute_gradients(self, v0, vk, ph0, phk):

        """i loops over examples in the minibatch"""


        def break_condition(i, v0, vk, ph0, phk, dW, db_h, db_v):
            return tf.less(i,k)[0]


        def body(i, v0, vk, ph0, phk, dW, db_h, db_v):
            v0_ = v0[i]
            ph0_ = ph0[i]
            vk_ = vk[i]
            phk_ = phk[i]

            v0_ = tf.reshape(v0_, [self.v_size, 1])
            ph0_ = tf.reshape(ph0_, [1, self.h_size])
            vk_ = tf.reshape(vk_, [self.v_size, 1])
            phk_ = tf.reshape(phk_, [1, self.h_size])

            """dW = v_0 (x) p(h0|v0)  - v_k (x) p(hk|vk)"""

            dW_ = tf.multiply(ph0_, v0_) - tf.multiply(phk_, vk_)
            dbh_ = ph0_ - phk_
            dbv_ = v0_ - vk_


            dbh_ = tf.reshape(dbh_, [self.h_size])
            dbv_ = tf.reshape(dbv_, [self.v_size])

            return [i+1, v0, vk, ph0, phk, tf.add(dbh, dbh_), tf.add(dbv, dbv_)]


        i=0
        k = tf.constant([self.batch_size])

        dW = tf.zeros((self.v_size, self.h_size))
        dbh = tf.zeros((self.h_size))
        dbv = tf.zeros((self.v_size))

        [i, v0, vk, ph0, phk, dW, db_h, db_v] = tf.while_loop(break_condition, loop_body, [i, v0, vk, ph0, phk, dW,dbh,dbv])


        dW /= self.batch_size
        dbh /= self.batch_size
        dbv /= self.batch_size

        return dW, dbh, dbv


    def optimize(self, v):
        v0, vk, ph0, phk, _ = self._gibbs_sample(v)
        dW, db_h, db_v = self._compute_gradients(v0, vk, ph0, phk)
        update_op = self._update_parameter(dW, db_hm, db_v)


        mask=tf.where(tf.less(v0,0.0),x=tf.zeros_like(v0),y=tf.ones_like(v0))
        bool_mask=tf.cast(tf.where(tf.less(v0,0.0),x=tf.zeros_like(v0),y=tf.ones_like(v0)), dtype=tf.bool)
        acc=tf.where(bool_mask,x=tf.abs(tf.subtract(v0,vk)),y=tf.zeros_like(v0))
        n_values=tf.reduce_sum(mask)
        acc=tf.subtract(1.0,tf.div(tf.reduce_sum(acc), n_values))

        return update_op, acc


    def _update_parameters(self, dW, db_h, db_v):
        update_op = [
                     tf.assign(self.W, tf.add(self.W, self.learning_rate*self.dW)),
                     tf.assign(self.bh, tf.add(self.bh, self.learning_rate*self.db_h)),
                     tf.assign(self.bv, tf.add(self.bv, self.learning_rate*self.db_v))
                    ]
        return update_op


    def inference(self, v):
        """v is set to as much information as we know. We then use this to make an estimate of h,
        then use that h to make an estimate of v.  This 'new' v will include estimates of vs we do not know """
        prob_h = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(v, self.W), self.bh))
        samp_h = self._bernoulli_sample(prob_h)

        prob_v = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(samp_h, tf.transpose(self.W, [1,0])), self.bv))
        samp_v = self._bernoulli_sample(prob_v)
        return samp_v
