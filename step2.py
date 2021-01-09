# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com


import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import sys, os, time, codecs, pdb

from utils.tf_funcs import *
from utils.prepare_data import *

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', 'data_combine/w2v_200.txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
## input struct ##
tf.app.flags.DEFINE_integer('max_sen_len', 30, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of tokens per documents')
## model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 200, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
tf.app.flags.DEFINE_integer('n_head', 4, 'number of heads of multi-head attention')
tf.app.flags.DEFINE_float('weight', 2.5, 'weight of weighted loss')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 5, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'P', 'RNN scope')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 128, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 0.8, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.flags.DEFINE_integer("display_step", 20,
                        """Number of steps to display log into TensorBoard (default: 20)""")



def build_model(word_embedding, pos1_embedding, pos2_embedding, x, sen_len, doc_len, keep_prob1, keep_prob2, related_dis1, related_dis2, y, pair, RNN = biLSTM):
    inputs = tf.nn.embedding_lookup(word_embedding, x)
    sen_len = tf.reshape(sen_len, [-1])
    
    inputs = tf.reshape(inputs, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    def get_s(inputs, name):
        with tf.name_scope('word_encode'):
            #inputs = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope+'word_layer' + name)
            inputs = tf.layers.dense(inputs, FLAGS.n_hidden, use_bias=True)
        with tf.name_scope('word_attention'):
            sh2 = inputs.get_shape().as_list()[-1]
            w1 = get_weight_varible('word_att_w1' + name, [sh2, sh2])
            b1 = get_weight_varible('word_att_b1' + name, [sh2])
            w2 = get_weight_varible('word_att_w2' + name, [sh2, 1])
            s = att_var(inputs,sen_len,w1,b1,w2)
        s = tf.reshape(s, [-1, FLAGS.max_doc_len, sh2])
        return s
    s = get_s(inputs, name='cause_word_encode')

    dis1 = tf.nn.embedding_lookup(pos1_embedding, related_dis1)
    dis2 = tf.nn.embedding_lookup(pos1_embedding, related_dis2)
    s = tf.concat([s, dis1, dis2], axis=-1)
    s = tf.layers.batch_normalization(s)

    def scaled_dot_product_attention(Q, K, V):
        d_k = Q.get_shape().as_list()[-1]
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        outputs /= d_k ** 0.5
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        outputs = tf.matmul(outputs, V)
        return outputs
    
    def multi_head_attention(query, key, value, num_heads, scope):
        d_k = query.get_shape().as_list()[-1]

        Q = tf.layers.dense(query, d_k*num_heads, use_bias=True)
        K = tf.layers.dense(key, d_k*num_heads, use_bias=True)
        V = tf.layers.dense(value, d_k*num_heads, use_bias=True)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        outputs = scaled_dot_product_attention(Q_, K_, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs = tf.layers.dense(outputs, d_k, use_bias=True)

        outputs += query
        outputs = tf.layers.batch_normalization(outputs)
        return outputs
    
    d_model = s.get_shape().as_list()[-1]

    emotion_index = pair[:, 0]
    cause_index = pair[:, 1]
    emotion_clause = tf.gather_nd(s, tf.stack((tf.range(tf.shape(s)[0], dtype=tf.int32), emotion_index), axis=1))
    cause_clause = tf.gather_nd(s, tf.stack((tf.range(tf.shape(s)[0], dtype=tf.int32), cause_index), axis=1))
    emotion_clause = tf.reshape(emotion_clause, shape=[-1, 1, d_model])
    cause_clause = tf.reshape(cause_clause, shape=[-1, 1, d_model])
    
    emotion_output = multi_head_attention(emotion_clause, s, s, num_heads=FLAGS.n_head, scope='emo')
    cause_output = multi_head_attention(cause_clause, s, s, num_heads=FLAGS.n_head, scope='cau')
    
    s = tf.stack([emotion_output, cause_output], axis=1)
    s = tf.reshape(s, [-1, 2*d_model])

    s1 = tf.nn.dropout(s, keep_prob=keep_prob2)
    w_pair = get_weight_varible('softmax_w_pair', [2*d_model, FLAGS.n_class])
    b_pair = get_weight_varible('softmax_b_pair', [FLAGS.n_class])
    pred_pair = tf.nn.softmax(tf.matmul(s1, w_pair) + b_pair)
        
    reg = tf.nn.l2_loss(w_pair) + tf.nn.l2_loss(b_pair)
    return pred_pair, reg

def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('pos_emb_dim-{}, num_heads-{}, weight-{}, n_hidden-{}, batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
        FLAGS.embedding_dim_pos, FLAGS.n_head, FLAGS.weight, FLAGS.n_hidden, FLAGS.batch_size,  FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data_new(x, sen_len, doc_len, keep_prob1, keep_prob2, related_dis1, related_dis2, y, pair, batch_size, test=False):
    for index in batch_index(len(y), batch_size, test):
        feed_list = [x[index], sen_len[index], doc_len[index], keep_prob1, keep_prob2, related_dis1[index], related_dis2[index], y[index], pair[index]]
        yield feed_list, len(index)

def run():
    save_dir = 'pair_data/{}/'.format(FLAGS.scope)
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if FLAGS.log_file_name:
        sys.stdout = open(save_dir + FLAGS.log_file_name, 'w')
    print_time()
    tf.reset_default_graph()
    # Model Code Block
    word_idx_rev, word_id_mapping, word_embedding, pos1_embedding, pos2_embedding = load_w2v_2nd(FLAGS.embedding_dim, FLAGS.embedding_dim_pos, 'data_combine/clause_keywords.csv', FLAGS.w2v_file)
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos1_embedding = tf.constant(pos1_embedding, dtype=tf.float32, name='pos1_embedding')
    pos2_embedding = tf.constant(pos2_embedding, dtype=tf.float32, name='pos2_embedding')

    print('build model...')
    
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    related_dis1 = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    related_dis2 = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
    pair = tf.placeholder(tf.int32, [None, 2])
    placeholders = [x, sen_len, doc_len, keep_prob1, keep_prob2, related_dis1, related_dis2, y, pair]
    
    
    pred_pair, reg = build_model(word_embedding, pos1_embedding, pos2_embedding, x, sen_len, doc_len, keep_prob1, keep_prob2, related_dis1, related_dis2, y, pair)
    loss_op = - tf.reduce_mean(y * tf.log(tf.clip_by_value(pred_pair, 1e-10, 1.0)) * [FLAGS.weight, 1]) + reg * FLAGS.l2_reg

    train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)
    
    true_y_op = tf.argmax(y, 1)
    pred_y_op = tf.argmax(pred_pair, 1)
    acc_op = tf.reduce_mean(tf.cast(tf.equal(true_y_op, pred_y_op), tf.float32))
    print('build model done!\n')
    
    # Training Code Block
    print_training_info()
    tf_config = tf.ConfigProto()  
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        keep_rate_list, acc_subtask_list, p_pair_list, r_pair_list, f1_pair_list = [], [], [], [], []
        o_p_pair_list, o_r_pair_list, o_f1_pair_list = [], [], []
        epoch_time_list = []
        
        for fold in range(1,11):
            sess.run(tf.global_variables_initializer())
            # train for one fold
            print('############# fold {} begin ###############'.format(fold))
            # Data Code Block
            train_file_name = 'fold{}_train.txt'.format(fold)
            test_file_name = 'fold{}_test.txt'.format(fold)
            tr_pair_id_all, tr_pair_id, tr_y, tr_x, tr_pair, tr_sen_len, tr_doc_len, tr_related_dis1, tr_related_dis2 = load_data_2nd_step_new(save_dir + train_file_name, word_id_mapping, max_sen_len = FLAGS.max_sen_len)
            te_pair_id_all, te_pair_id, te_y, te_x, te_pair, te_sen_len, te_doc_len, te_related_dis1, te_related_dis2 = load_data_2nd_step_new(save_dir + test_file_name, word_id_mapping, max_sen_len = FLAGS.max_sen_len)
            

            max_acc_subtask, max_f1 = [-1.]*2
            print('train docs: {}    test docs: {}'.format(len(tr_x), len(te_x)))
            for i in list(range(FLAGS.training_iter)):
                start_time, step = time.time(), 1
                # train
                print('########################### train ###########################')
                y_true = 0
                y_pred = 0
                for train, _ in get_batch_data_new(tr_x, tr_sen_len, tr_doc_len, FLAGS.keep_prob1, FLAGS.keep_prob2, tr_related_dis1, tr_related_dis2, tr_y, tr_pair, FLAGS.batch_size):
                    _, loss, pred_y, true_y, acc = sess.run(
                        [train_op, loss_op, pred_y_op, true_y_op, acc_op], feed_dict=dict(zip(placeholders, train)))
                    if step % FLAGS.display_step == 0:
                        print('step {}: train loss {:.4f} acc {:.4f}'.format(step, loss, acc))
                    step = step + 1
                    y_true += sum(true_y)
                    y_pred += sum(pred_y)
                print('label')
                print(y_true)
                print('pred')
                print(y_pred)

                # test
                print('########################### test ###########################')
                y_true = 0
                y_pred = 0
                test = [te_x, te_sen_len, te_doc_len, 1., 1., te_related_dis1, te_related_dis2, te_y, te_pair]
                loss, pred_y, true_y, acc = sess.run([loss_op, pred_y_op, true_y_op, acc_op], feed_dict=dict(zip(placeholders, test)))
                print('label')
                print(sum(true_y))
                print('pred')
                print(sum(pred_y))
                epoch_time = time.time()-start_time
                epoch_time_list.append(epoch_time)
                print('\nepoch {}: test loss {:.4f}, acc {:.4f}, cost time: {:.1f}s\n'.format(i, loss, acc, epoch_time))
                if acc > max_acc_subtask:
                    max_acc_subtask = acc
                print('max_acc_subtask: {:.4f} \n'.format(max_acc_subtask))
                
                # p, r, f1, o_p, o_r, o_f1, keep_rate = prf_2nd_step(te_pair_id_all, te_pair_id, pred_y, fold, save_dir)
                p, r, f1, o_p, o_r, o_f1, keep_rate = prf_2nd_step(te_pair_id_all, te_pair_id, pred_y)
                if f1 > max_f1:
                    max_keep_rate, max_p, max_r, max_f1 = keep_rate, p, r, f1
                print('original o_p {:.4f} o_r {:.4f} o_f1 {:.4f}'.format(o_p, o_r, o_f1))
                print('pair filter keep rate: {}'.format(keep_rate))
                print('test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))

                print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p, max_r, max_f1))
                

            print('Optimization Finished!\n')
            print('############# fold {} end ###############'.format(fold))
            # fold += 1
            acc_subtask_list.append(max_acc_subtask)
            keep_rate_list.append(max_keep_rate)
            p_pair_list.append(max_p)
            r_pair_list.append(max_r)
            f1_pair_list.append(max_f1)
            o_p_pair_list.append(o_p)
            o_r_pair_list.append(o_r)
            o_f1_pair_list.append(o_f1)
            
            print_training_info()
        all_results = [acc_subtask_list, keep_rate_list, p_pair_list, r_pair_list, f1_pair_list, o_p_pair_list, o_r_pair_list, o_f1_pair_list, epoch_time_list]
        acc_subtask, keep_rate, p_pair, r_pair, f1_pair, o_p_pair, o_r_pair, o_f1_pair, epoch_time_pair = map(lambda x: np.array(x).mean(), all_results)
        print('\nOriginal pair_predict: test f1 in 10 fold: {}'.format(np.array(o_f1_pair_list).reshape(-1,1)))
        print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(o_p_pair, o_r_pair, o_f1_pair))
        print('\nAverage keep_rate: {:.4f}\n'.format(keep_rate))
        print('\nFiltered pair_predict: test f1 in 10 fold: {}'.format(np.array(f1_pair_list).reshape(-1,1)))
        print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p_pair, r_pair, f1_pair))
        print('\naverage time : {}s'.format(epoch_time_pair))
        print_time()
        
     
def main(_):
    
    # FLAGS.log_file_name = 'step2.log'
    FLAGS.scope = 'P_emotion_weight_3.0'
    run()

    


if __name__ == '__main__':
    tf.app.run() 
