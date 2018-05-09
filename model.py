from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

class PIModel(object):
    def __init__(self, config, pretrained_embeddings, model_type):
        self.model_type = model_type
        self.config = config
        self.embeddings = tf.Variable(pretrained_embeddings, trainable=self.config.retrain_embeddings)
        self.add_placeholders()
        self.add_embeddings()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_train_op()

    def add_placeholders(self):
        self.prem_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_prem_len))
        self.prem_len_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.hyp_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_hyp_len))
        self.hyp_len_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.label_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        self.l2_placeholder = tf.placeholder(tf.float32, shape = ())
        self.learning_rate_placeholder = tf.placeholder(tf.float32, shape=(1,))
        self.weights_placeholder = tf.placeholder(tf.int32, shape=(None,))

    def create_feed_dict(self, prem_batch, prem_len, hyp_batch, hyp_len, dropout, l2 = None, learning_rate = None, label_batch=None, weights = None):
        feed_dict = {
            self.prem_placeholder: prem_batch,
            self.prem_len_placeholder: prem_len,
            self.hyp_placeholder: hyp_batch,
            self.hyp_len_placeholder: hyp_len,
            self.dropout_placeholder: dropout,
            self.l2_placeholder: l2
        }
        if label_batch is not None:
            feed_dict[self.label_placeholder] = label_batch
        if weights is not None:
            feed_dict[self.weights_placeholder] = weights
        if learning_rate is not None:
            feed_dict[self.learning_rate_placeholder] = learning_rate
        else:
            learning_rate = 0
        return feed_dict

    def add_embeddings(self):
        self.embed_prems = tf.nn.embedding_lookup(self.embeddings, self.prem_placeholder)
        self.embed_hyps = tf.nn.embedding_lookup(self.embeddings, self.hyp_placeholder)

    def add_seq2seq_prediction_op(self):
        initer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("prem"):
            prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.config.state_size), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
            new_prems, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems,\
                          sequence_length=self.prem_len_placeholder, dtype=tf.float32)
        with tf.variable_scope("hyp"):
            hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.config.state_size), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
            _, outputs = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps,\
                         sequence_length=self.hyp_len_placeholder, initial_state=prem_out)
        h = outputs
        if self.config.attention:
            Wy = tf.Variable(initer([1,1,self.config.state_size, self.config.state_size]))
            Wh = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            w =  tf.Variable(initer([1,1,self.config.state_size]))
            M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(outputs, Wh), 1))
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
            r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
            Wp = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            Wx= tf.Variable(initer([self.config.state_size, self.config.state_size]))
            h = tf.tanh(tf.matmul(r, Wp) + tf.matmul(outputs, Wx))

        Ws = tf.Variable(initer([self.config.state_size,3]))
        bs = tf.Variable(tf.zeros([1,3]) + 1e-3)
        self.logits = tf.matmul(h, Ws) + bs

    def add_prediction_op(self):
        print("MODEL TYPE:", self.model_type)
        xavier = tf.contrib.layers.xavier_initializer()

        # ingest premise with premise-RNN; initialize hypothesis-RNN with output of premise-RNN
        if self.model_type == 'seq2seq':
            self.add_seq2seq_prediction_op()

        # ingest hypothesis and premise with two different RNNs, then concatenate the outputs of each
        if self.model_type == 'siamese':
            with tf.variable_scope("prem-siamese"):
                prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.config.state_size), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                _, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems,\
                              sequence_length=self.prem_len_placeholder, dtype=tf.float32)
            with tf.variable_scope("hyp-siamese"):
                hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.config.state_size), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                _, hyp_out = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps,\
                             sequence_length=self.hyp_len_placeholder, dtype=tf.float32)

            prem_projection = tf.layers.dense(prem_out, self.config.state_size/2)
            hyp_projection = tf.layers.dense(hyp_out, self.config.state_size/2)

            representation = tf.layers.dense(
                                            tf.concat([prem_out, hyp_out], 1),
                                            self.config.state_size,
                                            activation=tf.nn.relu,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            )

            self.logits = tf.layers.dense(representation, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)

        # bag of words: average premise, average hypothesis, then concatenate
        if self.model_type == 'bow':
            prem_mean = tf.reduce_mean(self.embed_prems, axis=-1)
            hyp_mean = tf.reduce_mean(self.embed_hyps, axis=-1)

            prem_projection = tf.layers.dense(prem_mean, self.config.state_size/2)
            hyp_projection = tf.layers.dense(hyp_mean, self.config.state_size/2)

            representation = tf.layers.dense(
                                            tf.concat([prem_mean, hyp_mean], 1),
                                            self.config.state_size,
                                            activation=tf.nn.relu,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            )

            self.logits = tf.layers.dense(representation, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)



    def add_loss_op(self):
        beta = self.l2_placeholder
        reg = 0
        for v in tf.trainable_variables():
            reg = reg + tf.nn.l2_loss(v)
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.label_placeholder, logits=self.logits, weights = self.weights_placeholder) + beta*reg)

    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder[0])
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def optimize(self, sess, train_x, train_y, lr, l2):
        prem_batch, prem_len, hyp_batch, hyp_len, dropout = train_x
        label_batch = train_y
        d = dict()
        for i in range(3):
            d[i] = 0
        for i in train_y:
            d[i] += 1
        s = d[0] + d[1] + d[2]
        for i in d:
            if d[i] != 0:
                d[i] =float(s)/float(d[i])
            else:
                d[i] = 0
        weights_batch = []
        for i in train_y:
            weights_batch.append(d[i])
        input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len, dropout, l2, lr, label_batch, np.array(weights_batch))
        output_feed = [self.train_op, self.logits, self.loss]
        _, logits, loss = sess.run(output_feed, input_feed)
        return np.argmax(logits, axis=1), loss

    def validate(self, sess, valid_x, valid_y):
        prem_batch, prem_len, hyp_batch, hyp_len = valid_x
        label_batch = valid_y
        input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len,1, 0, [0], label_batch, [0])
        output_feed = [self.logits, self.loss]
        logits, loss = sess.run(output_feed, input_feed)
        return np.argmax(logits, axis=1), loss

    def predict(self, sess, test_x):
        prem_batch, prem_len, hyp_batch, hyp_len = test_x
        input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len, 1)
        output_feed = [self.logits]
        logits = sess.run(output_feed, input_feed)
        return np.argmax(logits[0], axis=1)

    def run_train_epoch(self, sess, dataset, lr, dropout, l2):
        print(np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))
        preds = []
        labels = []
        losses = 0.
        x = 0
        count = 0
        for prem, prem_len, hyp, hyp_len, label in dataset:
            pred, loss = self.optimize(sess, (prem, prem_len, hyp, hyp_len, dropout), label, lr, l2)
            preds.extend(pred)
            labels.extend(label)
            losses += loss * len(label)
            x += 1
            count +=1
        return preds, labels, losses / len(labels)

    def run_val_epoch(self, sess, dataset):
        preds = []
        labels = []
        losses = 0.
        for prem, prem_len, hyp, hyp_len, label in dataset:
            pred, loss = self.validate(sess, (prem, prem_len, hyp, hyp_len), label)
            preds.extend(pred)
            labels.extend(label)
            losses += loss * len(label)
        return preds, labels, losses / len(labels)

    def run_test_epoch(self, sess, dataset):
        preds = []
        labels = []
        losses = 0.
        for prem, prem_len, hyp, hyp_len, label in dataset:
            pred = self.predict(sess, (prem, prem_len, hyp, hyp_len))
            preds.extend(pred)
            labels.extend(label)
        return preds, labels
