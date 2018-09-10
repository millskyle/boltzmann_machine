from data.dataset import _get_test_data,_get_training_data
import tensorflow as tf
from rbm import RBM


tf.app.flags.DEFINE_string('tf_records_train_path', 'data/tf_records_1M/train/',
                           'Path of the training data.')

tf.app.flags.DEFINE_string('tf_records_test_path', 'data/tf_records_1M/train/',
                           'Path of the test data.')

tf.app.flags.DEFINE_integer('num_epoch', 1000,
                            'Number of training epochs.')

tf.app.flags.DEFINE_integer('batch_size', 32,
                            'Size of the training batch.')

tf.app.flags.DEFINE_float('learning_rate',1.0,
                          'Learning_Rate')

tf.app.flags.DEFINE_integer('v_size', 3952,
                            'Number of visible neurons (Number of movies the users rated.)')

tf.app.flags.DEFINE_integer('h_size', 200,
                            'Number of hidden neurons.')

tf.app.flags.DEFINE_integer('num_samples', 6039,
                            'Number of training samples (Number of users, who gave a rating).')

tf.app.flags.DEFINE_integer('k', 10,
                           'Number of iterations during gibbs samling.')

tf.app.flags.DEFINE_integer('eval_after',50,
                            'Evaluate model after number of iterations.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    '''Build the graph'''
    num_batches = int(FLAGS.num_samples / FLAGS.batch_size)
    train_data, train_data_infer = _get_training_data(FLAGS)
    test_data = _get_test_data(FLAGS)

    iter_train = train_data.make_initializable_iterator()
    iter_train_infer = train_data_infer.make_initializable_iterator()
    iter_test = test_data.make_initializable_iterator()

    x_train = iter_train.get_next()
    x_train_infer = iter_train_infer.get_next()
    x_test = iter_test.get_next()

    model = RBM(FLAGS, )

main(0)
