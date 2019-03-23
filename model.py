from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

class PIModel(object):
    def __init__(self, config, pretrained_embeddings, model_type):
        self.weights1256=config.weights1256
        self.length = None
        self.model_type = model_type
        if self.model_type[0:4] == "rntn":
            self.rntn = True
        else:
            self.rntn = False
        self.config = config
        if self.config.activationstring == "relu":
            self.config.activation = tf.nn.relu
        if self.config.activationstring == "tanh":
            self.config.activation = tf.nn.tanh
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
        self.label_placeholder1256 = tf.placeholder(tf.int32, shape=(None,12))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        self.l2_placeholder = tf.placeholder(tf.float32, shape = ())
        self.learning_rate_placeholder = tf.placeholder(tf.float32, shape=())

    def create_feed_dict(self, prem_batch, prem_len, hyp_batch, hyp_len, dropout, l2 = None, learning_rate = None, label_batch=None, label_batch1256=None):
        feed_dict = {
            self.prem_placeholder: prem_batch,
            self.prem_len_placeholder: prem_len,
            self.hyp_placeholder: hyp_batch,
            self.hyp_len_placeholder: hyp_len,
            self.dropout_placeholder: dropout
        }
        if l2 is not None:
            feed_dict[self.l2_placeholder] = l2
        if label_batch is not None:
            feed_dict[self.label_placeholder] = label_batch
        if label_batch1256 is not None:
            feed_dict[self.label_placeholder1256] = label_batch1256
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
        xavier = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("prem-siamese"):
            prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="premsiamese"), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
            new_prems, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems,\
                          sequence_length=self.prem_len_placeholder, dtype=tf.float32)
        with tf.variable_scope("hyp-siamese"):
            hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="hypsiamese"), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
            new_hyps, hyp_out = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps,\
                         sequence_length=self.hyp_len_placeholder, initial_state=prem_out)
        hyp_out = hyp_out.h
        prem_out = prem_out.h
        h = hyp_out
        if self.config.attention == "simple":
            Wy = tf.Variable(initer([1,1,self.config.state_size, self.config.state_size]))
            Wh = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            w =  tf.Variable(initer([1,1,self.config.state_size]))
            M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(hyp_out, Wh), 1))
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
            r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
            Wp = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            Wx= tf.Variable(initer([self.config.state_size, self.config.state_size]))
            h = tf.tanh(tf.matmul(r, Wp) + tf.matmul(hyp_out, Wx))
        if self.config.attention == "wordbyword":
            Wy = tf.Variable(initer([1,1,self.config.state_size, self.config.state_size]),name="Wy")
            Wh = tf.Variable(initer([self.config.state_size, self.config.state_size]),name="Wh")
            Wr = tf.Variable(initer([self.config.state_size, self.config.state_size]),name="Wr")
            w =  tf.Variable(initer([1,1,self.config.state_size]),name="w")
            M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(new_hyps[:,0,:], Wh), 1))
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
            r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
            Wt = tf.Variable(initer([self.config.state_size, self.config.state_size]),name="Wt")
            for i in range(1,9):
                M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(new_hyps[:,i,:], Wh), 1) + tf.expand_dims(tf.matmul(r, Wr), 1))
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
                r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1) + tf.tanh(tf.matmul(r, Wt))
            Wp = tf.Variable(initer([self.config.state_size, self.config.state_size]),name="Wp")
            Wx= tf.Variable(initer([self.config.state_size, self.config.state_size]),name="Wx")
            h = self.config.activation(tf.matmul(r, Wp) + tf.matmul(hyp_out, Wx))
        Ws1 = tf.Variable(initer([self.config.state_size,self.config.state_size]),name="Ws1")
        bs1 = tf.Variable(tf.zeros([1,self.config.state_size]) + 1e-3,name="bs1")
        h = self.config.activation(tf.matmul(h, Ws1) + bs1)
        Ws3 = tf.Variable(initer([self.config.state_size,self.config.state_size]),name="Ws3")
        bs3 = tf.Variable(tf.zeros([1,self.config.state_size]) + 1e-3,name="bs3")
        h = self.config.activation(tf.matmul(h, Ws3) + bs3)

        Ws2 = tf.Variable(initer([self.config.state_size,3]),name="Ws2")
        bs2 = tf.Variable(tf.zeros([1,3]) + 1e-3,name="bs2")
        self.logits9 = tf.matmul(h, Ws2) + bs2

        Ws2 = tf.Variable(initer([self.config.state_size,4]), name="one")
        bs2 = tf.Variable(tf.zeros([1,4]) + 1e-3)
        self.logits1 = tf.matmul(h, Ws2) + bs2

        Ws2 = tf.Variable(initer([self.config.state_size,4]), name="two")
        bs2 = tf.Variable(tf.zeros([1,4]) + 1e-3)
        self.logits2 = tf.matmul(h, Ws2) + bs2

        Ws2 = tf.Variable(initer([self.config.state_size,7]), name="five")
        bs2 = tf.Variable(tf.zeros([1,7]) + 1e-3, name="five")
        self.logits5 = tf.matmul(h, Ws2) + bs2

        Ws2 = tf.Variable(initer([self.config.state_size,7]), name="six")
        bs2 = tf.Variable(tf.zeros([1,7]) + 1e-3, name="six")
        self.logits6 = tf.matmul(h, Ws2) + bs2
        self.logits1256 = []
        reuse = False
        for i in [1,2,4,5,7,8]:
            with tf.variable_scope("prem-siamese"):
                prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="premsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                new_prems, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems[:,i:i+1,:],\
                              sequence_length=self.prem_len_placeholder, dtype=tf.float32)
            with tf.variable_scope("hyp-siamese"):
                hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="hypsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                new_hyps, hyp_out = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps[:,i:i+1,:],\
                             sequence_length=self.hyp_len_placeholder, dtype=tf.float32)
            hyp_out = hyp_out.h
            prem_out = prem_out.h
            h = hyp_out
            if self.config.attention == "simple":
                Wy = tf.Variable(initer([1,1,self.config.state_size, self.config.state_size]))
                Wh = tf.Variable(initer([self.config.state_size, self.config.state_size]))
                w =  tf.Variable(initer([1,1,self.config.state_size]))
                M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(hyp_out, Wh), 1))
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
                r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
                Wp = tf.Variable(initer([self.config.state_size, self.config.state_size]))
                Wx= tf.Variable(initer([self.config.state_size, self.config.state_size]))
                h = tf.tanh(tf.matmul(r, Wp) + tf.matmul(hyp_out, Wx))
            if self.config.attention == "wordbyword":
                M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(new_hyps[:,0,:], Wh), 1))
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
                r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
                for j in range(1):
                    M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(new_hyps[:,j,:], Wh), 1) + tf.expand_dims(tf.matmul(r, Wr), 1))
                    alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
                    r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1) + tf.tanh(tf.matmul(r, Wt))
                h = self.config.activation(tf.matmul(r, Wp) + tf.matmul(hyp_out, Wx))
            h = self.config.activation(tf.matmul(h, Ws1) + bs1)
            finalrep = self.config.activation(tf.matmul(h, Ws3) + bs3)
            self.logits1256.append(tf.layers.dense(finalrep, 4,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          reuse=reuse,
                                          name="one2"))
            reuse = True

        reuse = False
        for i in [1,4,7]:
            with tf.variable_scope("prem-siamese"):
                prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="premsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                new_prems, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems[:,i:i+2,:],\
                              sequence_length=self.prem_len_placeholder, dtype=tf.float32)
            with tf.variable_scope("hyp-siamese"):
                hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="hypsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                new_hyps, hyp_out = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps[:,i:i+2,:],\
                             sequence_length=self.hyp_len_placeholder, dtype=tf.float32)
            hyp_out = hyp_out.h
            prem_out = prem_out.h
            h = hyp_out
            if self.config.attention == "simple":
                Wy = tf.Variable(initer([1,1,self.config.state_size, self.config.state_size]))
                Wh = tf.Variable(initer([self.config.state_size, self.config.state_size]))
                w =  tf.Variable(initer([1,1,self.config.state_size]))
                M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(hyp_out, Wh), 1))
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
                r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
                Wp = tf.Variable(initer([self.config.state_size, self.config.state_size]))
                Wx= tf.Variable(initer([self.config.state_size, self.config.state_size]))
                h = tf.tanh(tf.matmul(r, Wp) + tf.matmul(hyp_out, Wx))
            if self.config.attention == "wordbyword":
                M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(new_hyps[:,0,:], Wh), 1))
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
                r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
                for j in range(2):
                    M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(new_hyps[:,j,:], Wh), 1) + tf.expand_dims(tf.matmul(r, Wr), 1))
                    alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
                    r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1) + tf.tanh(tf.matmul(r, Wt))
                h = self.config.activation(tf.matmul(r, Wp) + tf.matmul(hyp_out, Wx))
            h = self.config.activation(tf.matmul(h, Ws1) + bs1)
            finalrep = self.config.activation(tf.matmul(h, Ws3) + bs3)
            self.logits1256.append(tf.layers.dense(finalrep, 4,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          reuse=reuse,
                                          name="two2"))
            reuse = True
        with tf.variable_scope("prem-siamese"):
            prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="premsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
            new_prems, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems[:,4:,:],\
                          sequence_length=self.prem_len_placeholder, dtype=tf.float32)
        with tf.variable_scope("hyp-siamese"):
            hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="hypsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
            new_hyps, hyp_out = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps[:,4:,:],\
                         sequence_length=self.hyp_len_placeholder, dtype=tf.float32)
        hyp_out = hyp_out.h
        prem_out = prem_out.h
        h = hyp_out
        if self.config.attention == "simple":
            Wy = tf.Variable(initer([1,1,self.config.state_size, self.config.state_size]))
            Wh = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            w =  tf.Variable(initer([1,1,self.config.state_size]))
            M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(hyp_out, Wh), 1))
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
            r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
            Wp = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            Wx= tf.Variable(initer([self.config.state_size, self.config.state_size]))
            h = tf.tanh(tf.matmul(r, Wp) + tf.matmul(hyp_out, Wx))
        if self.config.attention == "wordbyword":
            M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(new_hyps[:,0,:], Wh), 1))
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
            r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
            for j in range(5):
                M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(new_hyps[:,j,:], Wh), 1) + tf.expand_dims(tf.matmul(r, Wr), 1))
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
                r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1) + tf.tanh(tf.matmul(r, Wt))
            h = self.config.activation(tf.matmul(r, Wp) + tf.matmul(hyp_out, Wx))
        h = self.config.activation(tf.matmul(h, Ws1) + bs1)
        finalrep = self.config.activation(tf.matmul(h, Ws3) + bs3)
        self.logits1256.append(tf.layers.dense(finalrep, 7,
                                      kernel_initializer=xavier,
                                      use_bias=True,
                                      name="five2"))
        with tf.variable_scope("prem-siamese"):
            prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="premsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
            new_prems, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems[:,3:,:],\
                          sequence_length=self.prem_len_placeholder, dtype=tf.float32)
        with tf.variable_scope("hyp-siamese"):
            hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="hypsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
            new_hyps, hyp_out = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps[:,3:,:],\
                         sequence_length=self.hyp_len_placeholder, dtype=tf.float32)
        hyp_out = hyp_out.h
        prem_out = prem_out.h
        h = hyp_out
        if self.config.attention == "simple":
            Wy = tf.Variable(initer([1,1,self.config.state_size, self.config.state_size]))
            Wh = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            w =  tf.Variable(initer([1,1,self.config.state_size]))
            M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(hyp_out, Wh), 1))
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
            r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
            Wp = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            Wx= tf.Variable(initer([self.config.state_size, self.config.state_size]))
            h = tf.tanh(tf.matmul(r, Wp) + tf.matmul(hyp_out, Wx))
        if self.config.attention == "wordbyword":
            M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(new_hyps[:,0,:], Wh), 1))
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
            r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
            for j in range(6):
                M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(new_hyps[:,j,:], Wh), 1) + tf.expand_dims(tf.matmul(r, Wr), 1))
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
                r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1) + tf.tanh(tf.matmul(r, Wt))
            h = self.config.activation(tf.matmul(r, Wp) + tf.matmul(hyp_out, Wx))
        h = self.config.activation(tf.matmul(h, Ws1) + bs1)
        finalrep = self.config.activation(tf.matmul(h, Ws3) + bs3)
        self.logits1256.append(tf.layers.dense(finalrep, 7,
                                      kernel_initializer=xavier,
                                      use_bias=True,
                                      name="six2"))
        self.logits1256.append(self.logits9)

    def LSTMcombine(self,children = None,input=None, size=None):
        if size is None:
            size = self.config.state_size
        if input == None:
            input = tf.Variable(tf.zeros([1,self.config.state_size]),trainable=False)
        if children == None:
            h1 = tf.Variable(tf.zeros([1,self.config.state_size]),trainable=False)
            c1 = tf.Variable(tf.zeros([1,self.config.state_size]),trainable=False)
            h2 = tf.Variable(tf.zeros([1,self.config.state_size]),trainable=False)
            c2 = tf.Variable(tf.zeros([1,self.config.state_size]),trainable=False)
        else:
            h1, c1 = children[0]
            h2, c2 = children[1]
        with tf.variable_scope("treeLSTM", reuse=True):
            Wi = tf.get_variable("LSTMWi")
            bi = tf.get_variable("LSTMbi")
            Wf=tf.get_variable("LSTMWf")
            bf=tf.get_variable("LSTMbf")
            Wo=tf.get_variable("LSTMWo")
            bo=tf.get_variable("LSTMbo")
            Wu=tf.get_variable("LSTMWu")
            bu=tf.get_variable("LSTMbu")
            Ui1=tf.get_variable("LSTMUi1")
            Ui2=tf.get_variable("LSTMUi2")
            Uo1=tf.get_variable("LSTMUo1")
            Uo2=tf.get_variable("LSTMUo2")
            Uu1=tf.get_variable("LSTMUu1")
            Uu2=tf.get_variable("LSTMUu2")
            Uf11=tf.get_variable("LSTMUf11")
            Uf12=tf.get_variable("LSTMUf12")
            Uf21=tf.get_variable("LSTMUf21")
            Uf22=tf.get_variable("LSTMUf22")
        i = tf.nn.sigmoid(tf.matmul(input, Wi) + tf.matmul(h1, Ui1)+tf.matmul(h2, Ui2) + bi)
        f1 = tf.nn.sigmoid(tf.matmul(input, Wf) + tf.matmul(h1, Uf11)+tf.matmul(h2, Uf12)+ bf)
        f2 = tf.nn.sigmoid(tf.matmul(input, Wf) + tf.matmul(h1, Uf21)+tf.matmul(h2, Uf22)+ bf)
        o = tf.nn.sigmoid(tf.matmul(input, Wo) + tf.matmul(h1, Uo1)+tf.matmul(h2, Uo1) + bo)
        u = tf.nn.tanh(tf.matmul(input, Wu) + tf.matmul(h1, Uu1)+tf.matmul(h2, Uu1) + bu)
        c = tf.multiply(i,u) + tf.multiply(f1, c1) + tf.multiply(f2,c2)
        h =  tf.multiply(o,tf.nn.tanh(c))
        return (h, c)

    def combine(self,stuff, name, reuse=True, size=None, input_sizes=None):
        if size is None:
            size = self.config.state_size
        if input_sizes is None:
            input_sizes = [self.config.state_size, self.config.state_size]
        xavier = tf.contrib.layers.xavier_initializer()
        if self.rntn and len(stuff) == 2:
            V= tf.Variable(xavier([input_sizes[0],size, input_sizes[1]]), name=name+"v")
            b= tf.Variable(tf.zeros([1,size]) + 1e-3, name=name+"b")
            W= tf.Variable(xavier([sum(input_sizes),size]), name=name+"w")
            return self.config.activation(tf.reduce_sum(tf.multiply(tf.tensordot(stuff[0], V, [[1], [0]]), tf.expand_dims(stuff[1], 1)))+ tf.matmul(tf.concat(stuff, 1), W) + b)
        return tf.layers.dense(
                                tf.concat(stuff, 1),
                                size,
                                activation=self.config.activation,
                                kernel_initializer=xavier,
                                use_bias=True,
                                name=name,
                                reuse=reuse
                                )

    def add_prediction_op(self):
        print("MODEL TYPE:", self.model_type)
        xavier = tf.contrib.layers.xavier_initializer()
        initer = tf.contrib.layers.xavier_initializer()

        # ingest premise with premise-RNN; initialize hypothesis-RNN with output of premise-RNN
        if self.model_type == 'seq2seq':
            self.add_seq2seq_prediction_op()

        # ingest hypothesis and premise with two different RNNs, then concatenate the outputs of each
        if self.model_type == 'siamese':
            with tf.variable_scope("prem-siamese"):
                prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="premsiamese"), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                _, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems,\
                              sequence_length=self.prem_len_placeholder, dtype=tf.float32)
            with tf.variable_scope("hyp-siamese"):
                hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="hypsiamese"), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                _, hyp_out = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps,\
                             sequence_length=self.hyp_len_placeholder, dtype=tf.float32)

            representation = tf.layers.dense(
                                            tf.concat([prem_out.h, hyp_out.h], 1),
                                            self.config.state_size,
                                            activation=self.config.activation,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            name="final"
                                            )
            representation2 = tf.layers.dense(
                                            representation,
                                            self.config.state_size,
                                            activation=self.config.activation,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            name = "final2"
                                            )

            self.logits9 = tf.layers.dense(representation2, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)
            self.logits1 = tf.layers.dense(representation2, 4,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="one")
            self.logits2 = tf.layers.dense(representation2, 4,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="two")
            self.logits5 = tf.layers.dense(representation2, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="five")
            self.logits6 = tf.layers.dense(representation2, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="six")
            self.logits1256 = []
            reuse = False
            for i in [1,2,4,5,7,8]:
                with tf.variable_scope("prem-siamese"):
                    prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="premsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                    _, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems[:,i:i+1,:],\
                                  sequence_length=self.prem_len_placeholder, dtype=tf.float32)
                with tf.variable_scope("hyp-siamese"):
                    hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="hypsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                    _, hyp_out = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps[:,i:i+1,:],\
                                 sequence_length=self.hyp_len_placeholder, dtype=tf.float32)
                representation = tf.layers.dense(
                                                tf.concat([prem_out.h, hyp_out.h], 1),
                                                self.config.state_size,
                                                activation=self.config.activation,
                                                kernel_initializer=xavier,
                                                use_bias=True,
                                                name="final",
                                                reuse=True
                                                )
                finalrep = tf.layers.dense(
                                                representation,
                                                self.config.state_size,
                                                activation=self.config.activation,
                                                kernel_initializer=xavier,
                                                use_bias=True,
                                                name = "final2",
                                                reuse=True
                                                )
                self.logits1256.append(tf.layers.dense(finalrep, 4,
                                              kernel_initializer=xavier,
                                              use_bias=True,
                                              reuse=reuse,
                                              name="one2"))
                reuse = True

            reuse = False
            for i in [1,4,7]:
                with tf.variable_scope("prem-siamese"):
                    prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="premsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                    _, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems[:,i:i+2,:],\
                                  sequence_length=self.prem_len_placeholder, dtype=tf.float32)
                with tf.variable_scope("hyp-siamese"):
                    hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="hypsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                    _, hyp_out = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps[:,i:i+2,:],\
                                 sequence_length=self.hyp_len_placeholder, dtype=tf.float32)
                representation = tf.layers.dense(
                                                tf.concat([prem_out.h, hyp_out.h], 1),
                                                self.config.state_size,
                                                activation=self.config.activation,
                                                kernel_initializer=xavier,
                                                use_bias=True,
                                                name="final",
                                                reuse=True
                                                )
                finalrep = tf.layers.dense(
                                                representation,
                                                self.config.state_size,
                                                activation=self.config.activation,
                                                kernel_initializer=xavier,
                                                use_bias=True,
                                                name = "final2",
                                                reuse=True
                                                )
                self.logits1256.append(tf.layers.dense(finalrep, 4,
                                              kernel_initializer=xavier,
                                              use_bias=True,
                                              reuse=reuse,
                                              name="two2"))
                reuse = True
            with tf.variable_scope("prem-siamese"):
                prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="premsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                _, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems[:,4:,:],\
                              sequence_length=self.prem_len_placeholder, dtype=tf.float32)
            with tf.variable_scope("hyp-siamese"):
                hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="hypsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                _, hyp_out = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps[:,4:,:],\
                             sequence_length=self.hyp_len_placeholder, dtype=tf.float32)
            representation = tf.layers.dense(
                                            tf.concat([prem_out.h, hyp_out.h], 1),
                                            self.config.state_size,
                                            activation=self.config.activation,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            name="final",
                                            reuse=True
                                            )
            finalrep = tf.layers.dense(
                                            representation,
                                            self.config.state_size,
                                            activation=self.config.activation,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            name = "final2",
                                            reuse=True
                                            )
            self.logits1256.append(tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="five2"))
            with tf.variable_scope("prem-siamese"):
                prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="premsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                _, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems[:,3:,:],\
                              sequence_length=self.prem_len_placeholder, dtype=tf.float32)
            with tf.variable_scope("hyp-siamese"):
                hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size, name="hypsiamese",reuse=True), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                _, hyp_out = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps[:,3:,:],\
                             sequence_length=self.hyp_len_placeholder, dtype=tf.float32)
            representation = tf.layers.dense(
                                            tf.concat([prem_out.h, hyp_out.h], 1),
                                            self.config.state_size,
                                            activation=self.config.activation,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            name="final",
                                            reuse=True
                                            )
            finalrep = tf.layers.dense(
                                            representation,
                                            self.config.state_size,
                                            activation=self.config.activation,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            name = "final2",
                                            reuse=True
                                            )
            self.logits1256.append(tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="six2"))
            self.logits1256.append(self.logits9)

        # bag of words: average premise, average hypothesis, then concatenate
        if self.model_type == 'bow':
            prem_mean = tf.reduce_mean(self.embed_prems, axis=-2)
            hyp_mean = tf.reduce_mean(self.embed_hyps, axis=-2)


            representation = tf.layers.dense(
                                            tf.concat([prem_mean, hyp_mean], 1),
                                            self.config.state_size,
                                            activation=self.config.activation,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            name="final"
                                            )
            representation = tf.layers.dense(
                                            representation,
                                            self.config.state_size,
                                            activation=self.config.activation,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            name="final2"
                                            )
            self.logits9 = tf.layers.dense(representation, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)
            self.logits1 = tf.layers.dense(representation, 4,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="one")
            self.logits2 = tf.layers.dense(representation, 4,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="two")
            self.logits5 = tf.layers.dense(representation, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="five")
            self.logits6 = tf.layers.dense(representation, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="six")
            self.logits1256 = []
            reuse = False
            for i in [1,2,4,5,7,8]:
                prem_mean = tf.reduce_mean(self.embed_prems[:,i:i+1,:], axis=-2)
                hyp_mean = tf.reduce_mean(self.embed_hyps[:,i:i+1,:], axis=-2)

                representation = tf.layers.dense(
                                                tf.concat([prem_mean, hyp_mean], 1),
                                                self.config.state_size,
                                                activation=self.config.activation,
                                                kernel_initializer=xavier,
                                                use_bias=True,
                                                name="final",
                                                reuse=True
                                                )
                finalrep = tf.layers.dense(
                                                representation,
                                                self.config.state_size,
                                                activation=self.config.activation,
                                                kernel_initializer=xavier,
                                                use_bias=True,
                                                name="final2",
                                                reuse=True
                                                )
                self.logits1256.append(tf.layers.dense(finalrep, 4,
                                              kernel_initializer=xavier,
                                              use_bias=True,
                                              reuse=reuse,
                                              name="one2"))
                reuse = True

            reuse = False
            for i in [1,4,7]:
                prem_mean = tf.reduce_mean(self.embed_prems[:,i:i+2,:], axis=-2)
                hyp_mean = tf.reduce_mean(self.embed_hyps[:,i:i+2,:], axis=-2)

                representation = tf.layers.dense(
                                                tf.concat([prem_mean, hyp_mean], 1),
                                                self.config.state_size,
                                                activation=self.config.activation,
                                                kernel_initializer=xavier,
                                                use_bias=True,
                                                name="final",
                                                reuse=True
                                                )
                finalrep = tf.layers.dense(
                                                representation,
                                                self.config.state_size,
                                                activation=self.config.activation,
                                                kernel_initializer=xavier,
                                                use_bias=True,
                                                name="final2",
                                                reuse=True
                                                )
                self.logits1256.append(tf.layers.dense(finalrep, 4,
                                              kernel_initializer=xavier,
                                              use_bias=True,
                                              reuse=reuse,
                                              name="two2"))
                reuse = True
            prem_mean = tf.reduce_mean(self.embed_prems[:,4:,:], axis=-2)
            hyp_mean = tf.reduce_mean(self.embed_hyps[:,4:,:], axis=-2)

            representation = tf.layers.dense(
                                            tf.concat([prem_mean, hyp_mean], 1),
                                            self.config.state_size,
                                            activation=self.config.activation,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            name="final",
                                            reuse=True
                                            )
            finalrep = tf.layers.dense(
                                            representation,
                                            self.config.state_size,
                                            activation=self.config.activation,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            name="final2",
                                            reuse=True
                                            )
            self.logits1256.append(tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="five2"))
            prem_mean = tf.reduce_mean(self.embed_prems[:,3:,:], axis=-2)
            hyp_mean = tf.reduce_mean(self.embed_hyps[:,3:,:], axis=-2)

            representation = tf.layers.dense(
                                            tf.concat([prem_mean, hyp_mean], 1),
                                            self.config.state_size,
                                            activation=self.config.activation,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            name="final",
                                            reuse=True
                                            )
            finalrep = tf.layers.dense(
                                            representation,
                                            self.config.state_size,
                                            activation=self.config.activation,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            name="final2",
                                            reuse=True
                                            )
            self.logits1256.append(tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="six2"))
            self.logits1256.append(self.logits9)
        if self.model_type == "restcomp":
            subjectd = self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"ycomp", reuse=False, size=16)
            subjectn = self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"zcomp",reuse=False, size=2)
            subjecta = self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])],"xcomp", reuse=False,size=4)
            neg = self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])],"vcomp",reuse=False,size=4)
            verb = self.combine([tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim])],"zcomp",size=2)
            adverb = self.combine([tf.reshape(self.embed_prems[:,6,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,6,:], [-1,self.config.vocab_dim])],"xcomp",size=4)
            objectd = self.combine([tf.reshape(self.embed_prems[:,7,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,7,:], [-1,self.config.vocab_dim])],"ycomp",size=16)
            objectn = self.combine([tf.reshape(self.embed_prems[:,8,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,8,:], [-1,self.config.vocab_dim])],"zcomp",size=2)
            objecta = self.combine([tf.reshape(self.embed_prems[:,9,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,9,:], [-1,self.config.vocab_dim])],"xcomp",size=4)
            subjectNP = self.combine([subjecta, subjectn],"acomp", reuse=False, size=4)
            objectNP = self.combine([objecta, objectn],"scomp", reuse=False, size=4)
            VP = self.combine([adverb, verb],"dcomp", reuse=False, size=4)
            objectDP1 = self.combine([objectd, objectNP],"fcomp", reuse=False, size=7)
            objectDP2 = self.combine([objectDP1, VP],"glcomp", reuse=False, size=7)
            negobjectDP = self.combine([neg, objectDP2],"afcomp", reuse=False, size=7)
            final = self.combine([subjectd, subjectNP,],"wefcomp", reuse=False, size=7)
            final2 = self.combine([final, negobjectDP],"fqecomp", reuse=False, size=7)
            self.logits = tf.layers.dense(final2, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)

        if self.model_type == "simpcomp" or self.model_type == "rntnsimpcomp":
            subjectd = self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"comp", reuse=False)
            subjectn = self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp")
            subjecta = self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])],"comp")
            neg = self.combine([tf.reshape(self.embed_prems[:,3,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,3,:], [-1,self.config.vocab_dim])],"comp")
            verb = self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])],"comp")
            adverb = self.combine([tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim])],"comp")
            objectd = self.combine([tf.reshape(self.embed_prems[:,6,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,6,:], [-1,self.config.vocab_dim])],"comp")
            objectn = self.combine([tf.reshape(self.embed_prems[:,7,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,7,:], [-1,self.config.vocab_dim])],"comp")
            objecta = self.combine([tf.reshape(self.embed_prems[:,8,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,8,:], [-1,self.config.vocab_dim])],"comp")
            subjectNP = self.combine([subjecta, subjectn],"comp")
            objectNP = self.combine([objecta, objectn],"comp")
            VP = self.combine([adverb, verb],"comp")
            objectDP1 = self.combine([objectd, objectNP],"comp")
            objectDP2 = self.combine([objectDP1, VP],"comp")
            negobjectDP = self.combine([neg, objectDP2],"comp")
            almostfinal = self.combine([subjectd, subjectNP,],"comp")
            final = self.combine([almostfinal, negobjectDP],"comp")
            finalrep = self.combine([final],"final", reuse=False)
            finalrep = self.combine([finalrep],"final2", reuse=False)
            self.logits9 = tf.layers.dense(finalrep, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)


            final= self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"comp")
            finalrep = self.combine([final],"final")
            finalrep = self.combine([finalrep],"final2")
            self.logits1 = tf.layers.dense(finalrep, 4,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="one")

            mod= self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"comp")
            arg= self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp")
            final= self.combine([mod, arg],"comp")
            finalrep = self.combine([final],"final")
            finalrep = self.combine([finalrep],"final2")
            self.logits2 = tf.layers.dense(finalrep, 4,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="two")

            det = self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"comp")
            mod1= self.combine([tf.reshape(self.embed_prems[:,3,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp")
            arg1= self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])],"comp")
            mod2= self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,3,:], [-1,self.config.vocab_dim])],"comp")
            arg2= self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])],"comp")
            rel1= self.combine([mod1, arg1],"comp")
            rel2= self.combine([mod2, arg2],"comp")
            DP1= self.combine([det, rel1],"comp")
            final= self.combine([DP1, rel2],"comp")
            finalrep = self.combine([final],"final")
            finalrep = self.combine([finalrep],"final2")
            self.logits5 = tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="five")

            neg = self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"comp")
            det = self.combine([tf.reshape(self.embed_prems[:,3,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp")
            mod1= self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])],"comp")
            arg1= self.combine([tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,3,:], [-1,self.config.vocab_dim])],"comp")
            mod2= self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])],"comp")
            arg2= self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim])],"comp")
            rel1 = self.combine([mod1, arg1],"comp")
            rel2= self.combine([mod2, arg2],"comp")
            DP1= self.combine([det, rel1],"comp")
            DP2= self.combine([DP1, rel2],"comp")
            final = self.combine([neg, DP2], "comp")
            finalrep = self.combine([final],"final")
            finalrep = self.combine([finalrep],"final2")
            self.logits6 = tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="six")
            self.logits1256 = []
            reuse = False
            for i in [1,2,4,5,7,8]:
                final = self.combine([tf.reshape(self.embed_prems[:,i,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,i,:], [-1,self.config.vocab_dim])],"comp")
                finalrep = self.combine([final],"final")
                finalrep = self.combine([finalrep],"final2")
                self.logits1256.append(tf.layers.dense(finalrep, 4,
                                              kernel_initializer=xavier,
                                              use_bias=True,
                                              reuse=reuse,
                                              name="one2"))
                reuse = True

            reuse = False
            for i in [1,4,7]:
                mod = self.combine([tf.reshape(self.embed_prems[:,i,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,i,:], [-1,self.config.vocab_dim])],"comp")
                arg = self.combine([tf.reshape(self.embed_prems[:,i+1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,i+1,:], [-1,self.config.vocab_dim])],"comp")
                final= self.combine([mod, arg],"comp")
                finalrep = self.combine([final],"final")
                finalrep = self.combine([finalrep],"final2")
                self.logits1256.append(tf.layers.dense(finalrep, 4,
                                              kernel_initializer=xavier,
                                              use_bias=True,
                                              reuse=reuse,
                                              name="two2"))
                reuse = True
            finalrep = self.combine([objectDP2],"final")
            finalrep = self.combine([finalrep],"final2")
            self.logits1256.append(tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="five2"))
            finalrep = self.combine([negobjectDP],"final")
            finalrep = self.combine([finalrep],"final2")
            self.logits1256.append(tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="six2"))
            self.logits1256.append(self.logits9)

        if self.model_type == "reallysimpcomp" or self.model_type == "rntnreallysimpcomp":
            subjectd = self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"compd", reuse=False)
            subjectn = self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp1", reuse=False)
            subjecta = self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])],"comp1")
            neg = self.combine([tf.reshape(self.embed_prems[:,3,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,3,:], [-1,self.config.vocab_dim])],"compneg", reuse=False)
            verb = self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])],"comp1")
            adverb = self.combine([tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim])],"comp1")
            objectd = self.combine([tf.reshape(self.embed_prems[:,6,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,6,:], [-1,self.config.vocab_dim])],"compd")
            objectn = self.combine([tf.reshape(self.embed_prems[:,7,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,7,:], [-1,self.config.vocab_dim])],"comp1")
            objecta = self.combine([tf.reshape(self.embed_prems[:,8,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,8,:], [-1,self.config.vocab_dim])],"comp1")
            subjectNP = self.combine([subjecta, subjectn],"comp2", reuse=False)
            objectNP = self.combine([objecta, objectn],"comp2")
            VP = self.combine([adverb, verb],"comp2")
            objectDP1 = self.combine([objectd, objectNP],"compobjDP1", reuse=False)
            objectDP2 = self.combine([objectDP1, VP],"compobjDP2", reuse=False)
            negobjectDP = self.combine([neg, objectDP2],"compnegobjDP", reuse=False)
            almostfinal = self.combine([subjectd, subjectNP,],"compclose", reuse=False)
            final = self.combine([almostfinal, negobjectDP],"compfinal", reuse=False)
            finalrep = self.combine([final],"final", reuse=False)
            finalrep = self.combine([finalrep],"final2", reuse=False)
            self.logits9 = tf.layers.dense(finalrep, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)

            finalrep= self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"comp1")
            self.logits1 = tf.layers.dense(finalrep, 4,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="one")

            mod= self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"comp1")
            arg= self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp1")
            finalrep= self.combine([mod, arg],"comp2")
            self.logits2 = tf.layers.dense(finalrep, 4,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="two")

            det = self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"compd")
            mod1= self.combine([tf.reshape(self.embed_prems[:,3,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp1")
            arg1= self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])],"comp1")
            mod2= self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,3,:], [-1,self.config.vocab_dim])],"comp1")
            arg2= self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])],"comp1")
            rel1= self.combine([mod1, arg1],"comp2")
            rel2= self.combine([mod2, arg2],"comp2")
            DP1= self.combine([det, rel1],"compobjDP1")
            finalrep= self.combine([DP1, rel2],"compobjDP2")
            self.logits5 = tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="five")

            neg = self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"compneg")
            det = self.combine([tf.reshape(self.embed_prems[:,3,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"compd")
            mod1= self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])],"comp1")
            arg1= self.combine([tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,3,:], [-1,self.config.vocab_dim])],"comp1")
            mod2= self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])],"comp1")
            arg2= self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim])],"comp1")
            rel1 = self.combine([mod1, arg1],"comp2")
            rel2= self.combine([mod2, arg2],"comp2")
            DP1= self.combine([det, rel1],"compobjDP1")
            DP2= self.combine([DP1, rel2],"compobjDP2")
            finalrep = self.combine([neg, DP2], "compnegobjDP")
            self.logits6 = tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="six")

        if self.model_type == "superreallysimpcomp" or self.model_type == "rntnsuperreallysimpcomp":
            subjectd = self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"compd", reuse=False, size=16)
            subjectn = self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp1", reuse=False, size = 4)
            subjecta = self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            neg = self.combine([tf.reshape(self.embed_prems[:,3,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,3,:], [-1,self.config.vocab_dim])],"compneg", reuse=False, size = 4)
            verb = self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            adverb = self.combine([tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            objectd = self.combine([tf.reshape(self.embed_prems[:,6,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,6,:], [-1,self.config.vocab_dim])],"compd", size = 16)
            objectn = self.combine([tf.reshape(self.embed_prems[:,7,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,7,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            objecta = self.combine([tf.reshape(self.embed_prems[:,8,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,8,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            subjectNP = self.combine([subjecta, subjectn],"comp2", reuse=False, size = 4, input_sizes=[4,4])
            objectNP = self.combine([objecta, objectn],"comp2", size = 4, input_sizes=[4,4])
            VP = self.combine([adverb, verb],"comp2", size = 4, input_sizes=[4,4])
            objectDP1 = self.combine([objectd, objectNP],"compobjDP1", reuse=False, size = 64, input_sizes=[16,4])
            objectDP2 = self.combine([objectDP1, VP],"compobjDP2", reuse=False, size = 7, input_sizes=[64,4])
            negobjectDP = self.combine([neg, objectDP2],"compnegobjDP", reuse=False, size = 7, input_sizes=[4,7])
            almostfinal = self.combine([subjectd, subjectNP,],"compclose", reuse=False, size=64,input_sizes=[16,4])
            finalrep = self.combine([almostfinal, negobjectDP],"compfinal", reuse=False, size =7,input_sizes=[64,7])
            self.logits9 = tf.layers.dense(finalrep, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)

            finalrep= self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            self.logits1 = finalrep

            mod= self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            arg= self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            finalrep= self.combine([mod, arg],"comp2", size=4, input_sizes=[4,4])
            self.logits2 = finalrep

            det = self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"compd", size=16)
            mod1= self.combine([tf.reshape(self.embed_prems[:,3,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            arg1= self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            mod2= self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,3,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            arg2= self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            rel1= self.combine([mod1, arg1],"comp2", size = 4, input_sizes=[4,4])
            rel2= self.combine([mod2, arg2],"comp2", size = 4, input_sizes=[4,4])
            DP1= self.combine([det, rel1],"compobjDP1", size = 64, input_sizes=[16,4])
            finalrep= self.combine([DP1, rel2],"compobjDP2", size = 7, input_sizes=[64,4])
            self.logits5 = finalrep

            neg = self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"compneg", size = 4)
            det = self.combine([tf.reshape(self.embed_prems[:,3,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"compd", size = 16)
            mod1= self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            arg1= self.combine([tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,3,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            mod2= self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            arg2= self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim])],"comp1", size = 4)
            rel1 = self.combine([mod1, arg1],"comp2", size = 4, input_sizes=[4,4])
            rel2= self.combine([mod2, arg2],"comp2", size = 4, input_sizes=[4,4])
            DP1= self.combine([det, rel1],"compobjDP1", size = 64, input_sizes=[16,4])
            DP2= self.combine([DP1, rel2],"compobjDP2", size = 7, input_sizes=[64,4])
            finalrep = self.combine([neg, DP2], "compnegobjDP", size = 7, input_sizes=[4,7])
            self.logits6 = finalrep

        if self.model_type == "LSTMsimpcomp":
            initer = tf.contrib.layers.xavier_initializer()
            biniter = tf.zeros_initializer
            with tf.variable_scope("treeLSTM"):
                Wi = tf.get_variable( "LSTMWi", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                bi = tf.get_variable("LSTMbi", shape=[1,self.config.state_size], initializer=biniter)
                Wf = tf.get_variable( "LSTMWf", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                bf = tf.get_variable("LSTMbf", shape=[1,self.config.state_size], initializer=biniter)
                Wo = tf.get_variable( "LSTMWo", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                bo = tf.get_variable("LSTMbo", shape=[1,self.config.state_size], initializer=biniter)
                Wu = tf.get_variable( "LSTMWu", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                bu = tf.get_variable("LSTMbu", shape=[1,self.config.state_size], initializer=biniter)
                Ui1 = tf.get_variable( "LSTMUi1", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Ui2 = tf.get_variable( "LSTMUi2", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uo1 = tf.get_variable( "LSTMUo1", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uo2 = tf.get_variable( "LSTMUo2", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uu1 = tf.get_variable( "LSTMUu1", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uu2 = tf.get_variable( "LSTMUu2", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uf11 = tf.get_variable( "LSTMUf11", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uf12 = tf.get_variable( "LSTMUf12", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uf21 = tf.get_variable( "LSTMUf21", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uf22 = tf.get_variable( "LSTMUf22", shape=[self.config.state_size,self.config.state_size], initializer=initer)
            psubjectd = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]))
            psubjectn = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]))
            psubjecta = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]))
            pneg = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]))
            pverb = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim]))
            padverb = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,6,:], [-1,self.config.vocab_dim]))
            pobjectd = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,7,:], [-1,self.config.vocab_dim]))
            pobjectn = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,8,:], [-1,self.config.vocab_dim]))
            pobjecta = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,9,:], [-1,self.config.vocab_dim]))
            hsubjectd = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim]))
            hsubjectn = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim]))
            hsubjecta = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim]))
            hneg = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim]))
            hverb = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim]))
            hadverb = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,6,:], [-1,self.config.vocab_dim]))
            hobjectd = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,7,:], [-1,self.config.vocab_dim]))
            hobjectn = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,8,:], [-1,self.config.vocab_dim]))
            hobjecta = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,9,:], [-1,self.config.vocab_dim]))
            neg = self.LSTMcombine(children =[pneg,hneg])
            verb = self.LSTMcombine(children =[pverb,hverb])
            adverb = self.LSTMcombine(children =[padverb,hadverb])
            subjectd = self.LSTMcombine(children =[psubjectd,hsubjectd])
            subjectn = self.LSTMcombine(children =[psubjectn,hsubjectn])
            subjecta = self.LSTMcombine(children =[psubjecta,hsubjecta])
            objectd = self.LSTMcombine(children =[pobjectd,hobjectd])
            objectn = self.LSTMcombine(children =[pobjectn,hobjectn])
            objecta = self.LSTMcombine(children =[pobjecta,hobjecta])
            subjectNP = self.LSTMcombine(children =[subjecta, subjectn])
            objectNP = self.LSTMcombine(children =[objecta, objectn])
            VP = self.LSTMcombine(children =[adverb, verb])
            objectDP1 = self.LSTMcombine(children =[objectd, objectNP])
            objectDP2 = self.LSTMcombine(children =[objectDP1, VP])
            negobjectDP = self.LSTMcombine(children =[neg, objectDP2])
            almostfinal = self.LSTMcombine(children=[subjectd, subjectNP,])
            final = self.LSTMcombine(children=[final, negobjectDP])
            finalrep = self.combine([almostfinal[0]],"final", reuse=False)
            self.logits = tf.layers.dense(finalrep, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)

        if self.model_type == "boolcomp":
            subjectd = self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"comp", reuse=False)
            subjectn = self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp")
            subjecta = self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])],"comp")
            neg = self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])],"comp")
            verb = self.combine([tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim])],"comp")
            adverb = self.combine([tf.reshape(self.embed_prems[:,6,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,6,:], [-1,self.config.vocab_dim])],"comp")
            objectd = self.combine([tf.reshape(self.embed_prems[:,7,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,7,:], [-1,self.config.vocab_dim])],"comp")
            objectn = self.combine([tf.reshape(self.embed_prems[:,8,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,8,:], [-1,self.config.vocab_dim])],"comp")
            objecta = self.combine([tf.reshape(self.embed_prems[:,9,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,9,:], [-1,self.config.vocab_dim])],"comp")
            subjectNP = self.combine([subjecta, subjectn],"comp")
            objectNP = self.combine([objecta, objectn],"comp")
            VP = self.combine([adverb, verb],"comp")
            objectDP1 = self.combine([objectd, objectNP],"comp")
            objectDP2 = self.combine([objectDP1, VP],"comp")
            negobjectDP = self.combine([neg, objectDP2],"comp")
            almostfinal = self.combine([subjectd, subjectNP,],"comp")
            final = self.combine([almostfinal, negobjectDP],"comp")
            subjectd2 = self.combine([tf.reshape(self.embed_prems[:,0 + 11,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0 + 11,:], [-1,self.config.vocab_dim])],"comp")
            subjectn2 = self.combine([tf.reshape(self.embed_prems[:,1 + 11,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1 + 11,:], [-1,self.config.vocab_dim])],"comp")
            subjecta2 = self.combine([tf.reshape(self.embed_prems[:,2 + 11,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2 + 11,:], [-1,self.config.vocab_dim])],"comp")
            neg2 = self.combine([tf.reshape(self.embed_prems[:,4 + 11,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4 + 11,:], [-1,self.config.vocab_dim])],"comp")
            verb2 = self.combine([tf.reshape(self.embed_prems[:,5 + 11,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,5 + 11,:], [-1,self.config.vocab_dim])],"comp")
            adverb2 = self.combine([tf.reshape(self.embed_prems[:,6 + 11,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,6 + 11,:], [-1,self.config.vocab_dim])],"comp")
            objectd2 = self.combine([tf.reshape(self.embed_prems[:,7 + 11,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,7 + 11,:], [-1,self.config.vocab_dim])],"comp")
            objectn2 = self.combine([tf.reshape(self.embed_prems[:,8 + 11,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,8 + 11,:], [-1,self.config.vocab_dim])],"comp")
            objecta2 = self.combine([tf.reshape(self.embed_prems[:,9 + 11,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,9 + 11,:], [-1,self.config.vocab_dim])],"comp")
            subjectNP2 = self.combine([subjecta2, subjectn2],"comp")
            objectNP2 = self.combine([objecta2, objectn2],"comp")
            VP2 = self.combine([adverb2, verb2],"comp")
            objectDP12 = self.combine([objectd2, objectNP2],"comp")
            objectDP22 = self.combine([objectDP12, VP2],"comp")
            negobjectDP2 = self.combine([neg2, objectDP22],"comp")
            almostfinal2 = self.combine([subjectd2, subjectNP2],"comp")
            final2 = self.combine([almostfinal2, negobjectDP2],"comp")
            #final2 = tf.nn.softmax(tf.layers.dense(final2, 3,
            #                              kernel_initializer=xavier,
            #                              use_bias=True))
            #final22 = tf.nn.softmax(tf.layers.dense(final22, 3,
            #                              kernel_initializer=xavier,
            #                              use_bias=True))
            conj = self.combine([tf.reshape(self.embed_prems[:,10,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,10,:], [-1,self.config.vocab_dim])],"comp")
            truefinal1 = self.combine([conj, final], "comp")
            truefinal2 = self.combine([truefinal1,final2], "comp")
            finalrep = self.combine([truefinal2], "final",reuse=False)
            self.logits = tf.layers.dense(finalrep, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)

        if self.model_type == "LSTMboolcomp":
            initer = tf.contrib.layers.xavier_initializer()
            biniter = tf.zeros_initializer
            with tf.variable_scope("treeLSTM"):
                Wi = tf.get_variable( "LSTMWi", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                bi = tf.get_variable("LSTMbi", shape=[1,self.config.state_size], initializer=biniter)
                Wf = tf.get_variable( "LSTMWf", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                bf = tf.get_variable("LSTMbf", shape=[1,self.config.state_size], initializer=biniter)
                Wo = tf.get_variable( "LSTMWo", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                bo = tf.get_variable("LSTMbo", shape=[1,self.config.state_size], initializer=biniter)
                Wu = tf.get_variable( "LSTMWu", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                bu = tf.get_variable("LSTMbu", shape=[1,self.config.state_size], initializer=biniter)
                Ui1 = tf.get_variable( "LSTMUi1", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Ui2 = tf.get_variable( "LSTMUi2", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uo1 = tf.get_variable( "LSTMUo1", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uo2 = tf.get_variable( "LSTMUo2", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uu1 = tf.get_variable( "LSTMUu1", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uu2 = tf.get_variable( "LSTMUu2", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uf11 = tf.get_variable( "LSTMUf11", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uf12 = tf.get_variable( "LSTMUf12", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uf21 = tf.get_variable( "LSTMUf21", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uf22 = tf.get_variable( "LSTMUf22", shape=[self.config.state_size,self.config.state_size], initializer=initer)
            psubjectd = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]))
            psubjectn = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]))
            psubjecta = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]))
            pneg = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]))
            pverb = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim]))
            padverb = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,6,:], [-1,self.config.vocab_dim]))
            pobjectd = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,7,:], [-1,self.config.vocab_dim]))
            pobjectn = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,8,:], [-1,self.config.vocab_dim]))
            pobjecta = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,9,:], [-1,self.config.vocab_dim]))
            hsubjectd = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim]))
            hsubjectn = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim]))
            hsubjecta = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim]))
            hneg = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim]))
            hverb = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim]))
            hadverb = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,6,:], [-1,self.config.vocab_dim]))
            hobjectd = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,7,:], [-1,self.config.vocab_dim]))
            hobjectn = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,8,:], [-1,self.config.vocab_dim]))
            hobjecta = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,9,:], [-1,self.config.vocab_dim]))
            neg = self.LSTMcombine(children =[pneg,hneg])
            verb = self.LSTMcombine(children =[pverb,hverb])
            adverb = self.LSTMcombine(children =[padverb,hadverb])
            subjectd = self.LSTMcombine(children =[psubjectd,hsubjectd])
            subjectn = self.LSTMcombine(children =[psubjectn,hsubjectn])
            subjecta = self.LSTMcombine(children =[psubjecta,hsubjecta])
            objectd = self.LSTMcombine(children =[pobjectd,hobjectd])
            objectn = self.LSTMcombine(children =[pobjectn,hobjectn])
            objecta = self.LSTMcombine(children =[pobjecta,hobjecta])
            subjectNP = self.LSTMcombine(children =[subjecta, subjectn])
            objectNP = self.LSTMcombine(children =[objecta, objectn])
            VP = self.LSTMcombine(children =[adverb, verb])
            objectDP1 = self.LSTMcombine(children =[objectd, objectNP])
            objectDP2 = self.LSTMcombine(children =[objectDP1, VP])
            negobjectDP = self.LSTMcombine(children =[neg, objectDP2])
            almostfinal = self.LSTMcombine(children=[subjectd, subjectNP,])
            final = self.LSTMcombine(children=[almostfinal, negobjectDP])
            psubjectd2 = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,0 + 11,:], [-1,self.config.vocab_dim]))
            psubjectn2 = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,1 + 11,:], [-1,self.config.vocab_dim]))
            psubjecta2 = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,2 + 11,:], [-1,self.config.vocab_dim]))
            pneg2 = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,4 + 11,:], [-1,self.config.vocab_dim]))
            pverb2 = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,5 + 11,:], [-1,self.config.vocab_dim]))
            padverb2 = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,6 + 11,:], [-1,self.config.vocab_dim]))
            pobjectd2 = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,7 + 11,:], [-1,self.config.vocab_dim]))
            pobjectn2 = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,8 + 11,:], [-1,self.config.vocab_dim]))
            pobjecta2 = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,9 + 11,:], [-1,self.config.vocab_dim]))
            hsubjectd2 = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,0 + 11,:], [-1,self.config.vocab_dim]))
            hsubjectn2 = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,1 + 11,:], [-1,self.config.vocab_dim]))
            hsubjecta2 = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,2 + 11,:], [-1,self.config.vocab_dim]))
            hneg2 = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,4 + 11,:], [-1,self.config.vocab_dim]))
            hverb2 = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,5 + 11,:], [-1,self.config.vocab_dim]))
            hadverb2 = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,6 + 11,:], [-1,self.config.vocab_dim]))
            hobjectd2 = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,7 + 11,:], [-1,self.config.vocab_dim]))
            hobjectn2 = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,8 + 11,:], [-1,self.config.vocab_dim]))
            hobjecta2 = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,9 + 11,:], [-1,self.config.vocab_dim]))
            neg2 = self.LSTMcombine(children =[pneg2,hneg2])
            verb2 = self.LSTMcombine(children =[pverb2,hverb2])
            adverb2 = self.LSTMcombine(children =[padverb2,hadverb2])
            subjectd2 = self.LSTMcombine(children =[psubjectd2,hsubjectd2])
            subjectn2 = self.LSTMcombine(children =[psubjectn2,hsubjectn2])
            subjecta2 = self.LSTMcombine(children =[psubjecta2,hsubjecta2])
            objectd2 = self.LSTMcombine(children =[pobjectd2,hobjectd2])
            objectn2 = self.LSTMcombine(children =[pobjectn2,hobjectn2])
            objecta2 = self.LSTMcombine(children =[pobjecta2,hobjecta2])
            subjectNP2 = self.LSTMcombine(children =[subjecta2, subjectn2])
            objectNP2 = self.LSTMcombine(children =[objecta2, objectn2])
            VP2 = self.LSTMcombine(children =[adverb2, verb2])
            objectDP12 = self.LSTMcombine(children =[objectd2, objectNP2])
            objectDP22 = self.LSTMcombine(children =[objectDP12, VP2])
            negobjectDP2 = self.LSTMcombine(children =[neg2, objectDP22])
            almostfinal2 = self.LSTMcombine(children=[subjectd2, subjectNP2])
            final2 = self.LSTMcombine(children=[almostfinal2, negobjectDP2])
            pconj = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,10,:], [-1,self.config.vocab_dim]))
            hconj = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,10,:], [-1,self.config.vocab_dim]))
            conj = self.LSTMcombine(children =[pconj,hconj])
            truefinal1 = self.LSTMcombine(children= [conj, final])
            truefinal2 = self.LSTMcombine(children= [truefinal1,final2])
            finalrep = self.combine([truefinal2], "final", reuse=False)
            self.logits = tf.layers.dense(finalrep, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)

        if self.model_type == "LSTMsepsimpcomp":
            initer = tf.contrib.layers.xavier_initializer()
            biniter = tf.zeros_initializer
            with tf.variable_scope("treeLSTM"):
                Wi = tf.get_variable( "LSTMWi", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                bi = tf.get_variable("LSTMbi", shape=[1,self.config.state_size], initializer=biniter)
                Wf = tf.get_variable( "LSTMWf", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                bf = tf.get_variable("LSTMbf", shape=[1,self.config.state_size], initializer=biniter)
                Wo = tf.get_variable( "LSTMWo", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                bo = tf.get_variable("LSTMbo", shape=[1,self.config.state_size], initializer=biniter)
                Wu = tf.get_variable( "LSTMWu", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                bu = tf.get_variable("LSTMbu", shape=[1,self.config.state_size], initializer=biniter)
                Ui1 = tf.get_variable( "LSTMUi1", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Ui2 = tf.get_variable( "LSTMUi2", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uo1 = tf.get_variable( "LSTMUo1", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uo2 = tf.get_variable( "LSTMUo2", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uu1 = tf.get_variable( "LSTMUu1", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uu2 = tf.get_variable( "LSTMUu2", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uf11 = tf.get_variable( "LSTMUf11", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uf12 = tf.get_variable( "LSTMUf12", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uf21 = tf.get_variable( "LSTMUf21", shape=[self.config.state_size,self.config.state_size], initializer=initer)
                Uf22 = tf.get_variable( "LSTMUf22", shape=[self.config.state_size,self.config.state_size], initializer=initer)
            psubjectd = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]))
            psubjectn = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]))
            psubjecta = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]))
            pneg = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]))
            pverb = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim]))
            padverb = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,6,:], [-1,self.config.vocab_dim]))
            pobjectd = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,7,:], [-1,self.config.vocab_dim]))
            pobjectn = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,8,:], [-1,self.config.vocab_dim]))
            pobjecta = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,9,:], [-1,self.config.vocab_dim]))
            psubjectNP = self.LSTMcombine(children=[psubjecta, psubjectn])
            pobjectNP = self.LSTMcombine(children=[pobjecta, pobjectn])
            pVP = self.LSTMcombine(children=[padverb, pverb])
            pobjectDP1 = self.LSTMcombine(children=[pobjectd, pobjectNP])
            pobjectDP2 = self.LSTMcombine(children=[pobjectDP1, pVP])
            pnegobjectDP = self.LSTMcombine(children=[pneg, pobjectDP2])
            palmostfinal = self.LSTMcombine(children=[psubjectd, psubjectNP,])
            pfinal = self.LSTMcombine(children=[palmostfinal, pnegobjectDP])
            hsubjectd = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim]))
            hsubjectn = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim]))
            hsubjecta = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim]))
            hneg = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim]))
            hverb = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim]))
            hadverb = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,6,:], [-1,self.config.vocab_dim]))
            hobjectd = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,7,:], [-1,self.config.vocab_dim]))
            hobjectn = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,8,:], [-1,self.config.vocab_dim]))
            hobjecta = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,9,:], [-1,self.config.vocab_dim]))
            hsubjectNP = self.LSTMcombine(children=[hsubjecta, hsubjectn])
            hobjectNP = self.LSTMcombine(children=[hobjecta, hobjectn])
            hVP = self.LSTMcombine(children=[hadverb, hverb])
            hobjectDP1 = self.LSTMcombine(children=[hobjectd, hobjectNP])
            hobjectDP2 = self.LSTMcombine(children=[hobjectDP1, hVP])
            hnegobjectDP = self.LSTMcombine(children=[hneg, hobjectDP2])
            halmostfinal = self.LSTMcombine(children=[hsubjectd, hsubjectNP,])
            hfinal = self.LSTMcombine(children=[halmostfinal, hnegobjectDP])
            final = self.combine([pfinal[0], hfinal[0]], "final", reuse=False)
            finalrep = self.combine([final],"final2", reuse=False)
            self.logits = tf.layers.dense(final2, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)

        if self.model_type == "sepsimpcomp":
            psubjectd = tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim])
            psubjectn = tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim])
            psubjecta = tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim])
            pneg = tf.reshape(self.embed_prems[:,3,:], [-1,self.config.vocab_dim])
            pverb = tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim])
            padverb = tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim])
            pobjectd = tf.reshape(self.embed_prems[:,6,:], [-1,self.config.vocab_dim])
            pobjectn = tf.reshape(self.embed_prems[:,7,:], [-1,self.config.vocab_dim])
            pobjecta = tf.reshape(self.embed_prems[:,8,:], [-1,self.config.vocab_dim])
            psubjectNP = self.combine([psubjecta, psubjectn],"comp",reuse=False)
            pobjectNP = self.combine([pobjecta, pobjectn],"comp")
            pVP = self.combine([padverb, pverb],"comp")
            pobjectDP1 = self.combine([pobjectd, pobjectNP],"comp")
            pobjectDP2 = self.combine([pobjectDP1, pVP],"comp")
            pnegobjectDP = self.combine([pneg, pobjectDP2],"comp")
            palmostfinal = self.combine([psubjectd, psubjectNP,],"comp")
            pfinal = self.combine([palmostfinal, pnegobjectDP],"comp")
            hsubjectd = tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])
            hsubjectn = tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])
            hsubjecta = tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])
            hneg = tf.reshape(self.embed_hyps[:,3,:], [-1,self.config.vocab_dim])
            hverb = tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])
            hadverb = tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim])
            hobjectd = tf.reshape(self.embed_hyps[:,6,:], [-1,self.config.vocab_dim])
            hobjectn = tf.reshape(self.embed_hyps[:,7,:], [-1,self.config.vocab_dim])
            hobjecta = tf.reshape(self.embed_hyps[:,8,:], [-1,self.config.vocab_dim])
            hsubjectNP = self.combine([hsubjecta, hsubjectn],"comp")
            hobjectNP = self.combine([hobjecta, hobjectn],"comp")
            hVP = self.combine([hadverb, hverb],"comp")
            hobjectDP1 = self.combine([hobjectd, hobjectNP],"comp")
            hobjectDP2 = self.combine([hobjectDP1, hVP],"comp")
            hnegobjectDP = self.combine([hneg, hobjectDP2],"comp")
            halmostfinal = self.combine([hsubjectd, hsubjectNP,],"comp")
            hfinal = self.combine([halmostfinal, hnegobjectDP],"comp")
            final = self.combine([pfinal, hfinal], "final", reuse=False)
            finalrep = self.combine([final], "final2", reuse=False)
            #premise_nodes = [psubjectd,psubjecta,psubjectn,pneg, padverb, pverb, pobjectd,pobjecta,pobjectn,psubjectNP, pobjectNP, pVP, pobjectDP1, pobjectDP2,pnegobjectDP, pfinal, pfinal2]
            #premise_nodes = [tf.expand_dims(x,1) for x in premise_nodes]
            #premise_nodes = tf.concat(premise_nodes,1)
            #hypothesis_nodes = [hsubjectd,hsubjecta,hsubjectn,hneg, hadverb, hverb, hobjectd,hobjecta,hobjectn,hsubjectNP, hobjectNP, hVP, hobjectDP1, hobjectDP2,hnegobjectDP, hfinal, hfinal2]
            #hypothesis_nodes = [tf.expand_dims(x,1) for x in hypothesis_nodes]
            #hypothesis_nodes = tf.concat(hypothesis_nodes,1)
            if self.config.attention == "wordbyword":
                Wy = tf.Variable(initer([1,1,self.config.state_size, self.config.state_size]))
                Wh = tf.Variable(initer([self.config.state_size, self.config.state_size]))
                Wr = tf.Variable(initer([self.config.state_size, self.config.state_size]))
                w =  tf.Variable(initer([1,1,self.config.state_size]))
                M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(premise_nodes,3)), 3) + tf.expand_dims(tf.matmul(hypothesis_nodes[:,0,:], Wh), 1))
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
                r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), premise_nodes), 1)
                Wt = tf.Variable(initer([self.config.state_size, self.config.state_size]))
                for i in range(1,17):
                    M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(premise_nodes,3)), 3) + tf.expand_dims(tf.matmul(hypothesis_nodes[:,i,:], Wh), 1) + tf.expand_dims(tf.matmul(r, Wr), 1))
                    alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
                    r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), premise_nodes), 1) +tf.tanh(tf.matmul(r, Wt))
                Wp = tf.Variable(initer([self.config.state_size, self.config.state_size]))
                Wx= tf.Variable(initer([self.config.state_size, self.config.state_size]))
                finalrep = tf.tanh(tf.matmul(r, Wp) + tf.matmul(finalrep, Wx))
            self.logits9 = tf.layers.dense(finalrep, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)

            final= self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])],"final")
            finalrep = self.combine([final], "final2")
            self.logits1 = tf.layers.dense(finalrep, 4,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="one")

            premrep= self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim])],"comp")
            hyprep= self.combine([tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp")
            final= self.combine([premrep, hyprep],"final")
            finalrep = self.combine([final], "final2")
            self.logits2 = tf.layers.dense(finalrep, 4,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="two")

            prel1= self.combine([tf.reshape(self.embed_prems[:,3,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim])],"comp")
            prel2= self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim])],"comp")
            pDP1= self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim]), prel1],"comp")
            premrep= self.combine([pDP1, prel2],"comp")
            hrel1= self.combine([tf.reshape(self.embed_hyps[:,3,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])],"comp")
            hrel2= self.combine([tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])],"comp")
            hDP1= self.combine([tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim]), hrel1],"comp")
            hyprep= self.combine([hDP1, hrel2],"comp")
            final= self.combine([premrep, hyprep],"final")
            finalrep = self.combine([final], "final2")
            self.logits5 = tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="five")

            prel1= self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim])],"comp")
            prel2= self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim])],"comp")
            pDP1 = self.combine([tf.reshape(self.embed_prems[:,3,:], [-1,self.config.vocab_dim]), prel1],"comp")
            pDP2 = self.combine([pDP1, prel2],"comp")
            pneg= self.combine([tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim]), pDP2], "comp")
            hrel1= self.combine([tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim])],"comp")
            hrel2= self.combine([tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])],"comp")
            hDP1= self.combine([tf.reshape(self.embed_hyps[:,3,:], [-1,self.config.vocab_dim]), hrel1],"comp")
            hDP2= self.combine([hDP1, hrel2],"comp")
            hneg= self.combine([tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim]), hDP2], "comp")
            final = self.combine([premrep, hyprep],"final")
            finalrep = self.combine([final], "final2")
            self.logits6 = tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="six")
            self.logits1256 = []
            reuse = False
            for i in [1,2,4,5,7,8]:
                final= self.combine([tf.reshape(self.embed_prems[:,i,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,i,:], [-1,self.config.vocab_dim])],"final")
                finalrep = self.combine([final], "final2")
                self.logits1256.append(tf.layers.dense(finalrep, 4,
                                              kernel_initializer=xavier,
                                              use_bias=True,
                                              reuse=reuse,
                                              name="one2"))
                reuse = True

            reuse = False
            for i in [1,4,7]:
                premrep= self.combine([tf.reshape(self.embed_prems[:,i,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_prems[:,i+1,:], [-1,self.config.vocab_dim])],"comp")
                hyprep= self.combine([tf.reshape(self.embed_hyps[:,i,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,i+1,:], [-1,self.config.vocab_dim])],"comp")
                final= self.combine([premrep, hyprep],"final")
                finalrep = self.combine([final], "final2")
                self.logits1256.append(tf.layers.dense(finalrep, 4,
                                              kernel_initializer=xavier,
                                              use_bias=True,
                                              reuse=reuse,
                                              name="two2"))
                reuse = True
            finalrep = self.combine([pobjectDP2,hobjectDP2],"final")
            finalrep = self.combine([finalrep],"final2")
            self.logits1256.append(tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="five2"))
            finalrep = self.combine([pnegobjectDP,hnegobjectDP],"final")
            finalrep = self.combine([finalrep],"final2")
            self.logits1256.append(tf.layers.dense(finalrep, 7,
                                          kernel_initializer=xavier,
                                          use_bias=True,
                                          name="six2"))
            self.logits1256.append(self.logits9)

        if self.model_type == "sepboolcomp":
            psubjectd = tf.reshape(self.embed_prems[:,0,:], [-1,self.config.vocab_dim])
            psubjectn = tf.reshape(self.embed_prems[:,1,:], [-1,self.config.vocab_dim])
            psubjecta = tf.reshape(self.embed_prems[:,2,:], [-1,self.config.vocab_dim])
            pneg = tf.reshape(self.embed_prems[:,4,:], [-1,self.config.vocab_dim])
            pverb = tf.reshape(self.embed_prems[:,5,:], [-1,self.config.vocab_dim])
            padverb = tf.reshape(self.embed_prems[:,6,:], [-1,self.config.vocab_dim])
            pobjectd = tf.reshape(self.embed_prems[:,7,:], [-1,self.config.vocab_dim])
            pobjectn = tf.reshape(self.embed_prems[:,8,:], [-1,self.config.vocab_dim])
            pobjecta = tf.reshape(self.embed_prems[:,9,:], [-1,self.config.vocab_dim])
            psubjectNP = self.combine([psubjecta, psubjectn],"comp",reuse=False)
            pobjectNP = self.combine([pobjecta, pobjectn],"comp")
            pVP = self.combine([padverb, pverb],"comp")
            pobjectDP1 = self.combine([pobjectd, pobjectNP],"comp")
            pobjectDP2 = self.combine([pobjectDP1, pVP],"comp")
            pnegobjectDP = self.combine([pneg, pobjectDP2],"comp")
            palmostfinal = self.combine([psubjectd, psubjectNP,],"comp")
            pfinal = self.combine([palmostfinal, pnegobjectDP],"comp")
            psubjectd2 = tf.reshape(self.embed_prems[:,0+11,:], [-1,self.config.vocab_dim])
            psubjectn2 = tf.reshape(self.embed_prems[:,1+11,:], [-1,self.config.vocab_dim])
            psubjecta2 = tf.reshape(self.embed_prems[:,2+11,:], [-1,self.config.vocab_dim])
            pneg2 = tf.reshape(self.embed_prems[:,4+11,:], [-1,self.config.vocab_dim])
            pverb2 = tf.reshape(self.embed_prems[:,5+11,:], [-1,self.config.vocab_dim])
            padverb2 = tf.reshape(self.embed_prems[:,6+11,:], [-1,self.config.vocab_dim])
            pobjectd2 = tf.reshape(self.embed_prems[:,7+11,:], [-1,self.config.vocab_dim])
            pobjectn2 = tf.reshape(self.embed_prems[:,8+11,:], [-1,self.config.vocab_dim])
            pobjecta2 = tf.reshape(self.embed_prems[:,9+11,:], [-1,self.config.vocab_dim])
            psubjectNP2 = self.combine([psubjecta2, psubjectn2],"comp")
            pobjectNP2 = self.combine([pobjecta2, pobjectn2],"comp")
            pVP2 = self.combine([padverb2, pverb2],"comp")
            pobjectDP12 = self.combine([pobjectd2, pobjectNP2],"comp")
            pobjectDP22 = self.combine([pobjectDP12, pVP2],"comp")
            pnegobjectDP2 = self.combine([pneg2, pobjectDP22],"comp")
            palmostfinal2 = self.combine([psubjectd2, psubjectNP2,],"comp")
            pfinal2 = self.combine([palmostfinal2, pnegobjectDP2],"comp")
            pconj = self.combine([tf.reshape(self.embed_prems[:,10,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,10,:], [-1,self.config.vocab_dim])],"comp")
            ptruefinal1 = self.combine([pconj, pfinal], "comp")
            ptruefinal2 = self.combine([ptruefinal1,pfinal2], "comp")
            hsubjectd = tf.reshape(self.embed_hyps[:,0,:], [-1,self.config.vocab_dim])
            hsubjectn = tf.reshape(self.embed_hyps[:,1,:], [-1,self.config.vocab_dim])
            hsubjecta = tf.reshape(self.embed_hyps[:,2,:], [-1,self.config.vocab_dim])
            hneg = tf.reshape(self.embed_hyps[:,4,:], [-1,self.config.vocab_dim])
            hverb = tf.reshape(self.embed_hyps[:,5,:], [-1,self.config.vocab_dim])
            hadverb = tf.reshape(self.embed_hyps[:,6,:], [-1,self.config.vocab_dim])
            hobjectd = tf.reshape(self.embed_hyps[:,7,:], [-1,self.config.vocab_dim])
            hobjectn = tf.reshape(self.embed_hyps[:,8,:], [-1,self.config.vocab_dim])
            hobjecta = tf.reshape(self.embed_hyps[:,9,:], [-1,self.config.vocab_dim])
            hsubjectNP = self.combine([hsubjecta, hsubjectn],"comp2", reuse=False)
            hobjectNP = self.combine([hobjecta, hobjectn],"comp2")
            hVP = self.combine([hadverb, hverb],"comp2")
            hobjectDP1 = self.combine([hobjectd, hobjectNP],"comp2")
            hobjectDP2 = self.combine([hobjectDP1, hVP],"comp2")
            hnegobjectDP = self.combine([hneg, hobjectDP2],"comp2")
            halmostfinal = self.combine([hsubjectd, hsubjectNP,],"comp2")
            hfinal = self.combine([halmostfinal, hnegobjectDP],"comp2")
            hsubjectd2 = tf.reshape(self.embed_hyps[:,0+11,:], [-1,self.config.vocab_dim])
            hsubjectn2 = tf.reshape(self.embed_hyps[:,1+11,:], [-1,self.config.vocab_dim])
            hsubjecta2 = tf.reshape(self.embed_hyps[:,2+11,:], [-1,self.config.vocab_dim])
            hneg2 = tf.reshape(self.embed_hyps[:,4+11,:], [-1,self.config.vocab_dim])
            hverb2 = tf.reshape(self.embed_hyps[:,5+11,:], [-1,self.config.vocab_dim])
            hadverb2 = tf.reshape(self.embed_hyps[:,6+11,:], [-1,self.config.vocab_dim])
            hobjectd2 = tf.reshape(self.embed_hyps[:,7+11,:], [-1,self.config.vocab_dim])
            hobjectn2 = tf.reshape(self.embed_hyps[:,8+11,:], [-1,self.config.vocab_dim])
            hobjecta2 = tf.reshape(self.embed_hyps[:,9+11,:], [-1,self.config.vocab_dim])
            hsubjectNP2 = self.combine([hsubjecta2, hsubjectn2],"comp2")
            hobjectNP2 = self.combine([hobjecta2, hobjectn2],"comp2")
            hVP2 = self.combine([hadverb2, hverb2],"comp2")
            hobjectDP12 = self.combine([hobjectd2, hobjectNP2],"comp2")
            hobjectDP22 = self.combine([hobjectDP12, hVP2],"comp2")
            hnegobjectDP2 = self.combine([hneg2, hobjectDP22],"comp2")
            halmostfinal2 = self.combine([hsubjectd2, hsubjectNP2,],"comp2")
            hfinal2 = self.combine([halmostfinal2, hnegobjectDP2],"comp2")
            hconj = self.combine([tf.reshape(self.embed_prems[:,10,:], [-1,self.config.vocab_dim]), tf.reshape(self.embed_hyps[:,10,:], [-1,self.config.vocab_dim])],"comp")
            htruefinal1 = self.combine([hconj, hfinal], "comp2")
            htruefinal2 = self.combine([htruefinal1,hfinal2], "comp2")
            final = self.combine([ptruefinal2, htruefinal2], "final", reuse=False)
            final2 = self.combine([final], "final2", reuse=False)
            self.logits = tf.layers.dense(final2, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)




    def add_loss_op(self):
        if True:
            beta = self.l2_placeholder
            reg = 0
            for v in tf.trainable_variables():
                reg = reg + tf.nn.l2_loss(v)
            self.loss1256 = 0
            for i in range(6):
                self.loss1256 += tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.label_placeholder1256[:,i], logits=self.logits1256[i]))*self.weights1256[0]*(1/6)
            for i in range(6,9):
                self.loss1256 += tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.label_placeholder1256[:,i], logits=self.logits1256[i]))*self.weights1256[1]*(1/3)
            self.loss1256 += tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.label_placeholder1256[:,9], logits=self.logits1256[9]))*self.weights1256[2]
            self.loss1256 += tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.label_placeholder1256[:,10], logits=self.logits1256[10]))*self.weights1256[3]
            self.loss1256 += tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.label_placeholder1256[:,11], logits=self.logits1256[11]))*self.weights1256[4]
            self.loss1256 += beta*reg
        beta = self.l2_placeholder
        reg = 0
        for v in tf.trainable_variables():
            reg = reg + tf.nn.l2_loss(v)
        self.loss1 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.label_placeholder, logits=self.logits1) + beta*reg)
        beta = self.l2_placeholder
        reg = 0
        for v in tf.trainable_variables():
            reg = reg + tf.nn.l2_loss(v)
        self.loss2 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.label_placeholder, logits=self.logits2) + beta*reg)
        beta = self.l2_placeholder
        reg = 0
        for v in tf.trainable_variables():
            reg = reg + tf.nn.l2_loss(v)
        self.loss5 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.label_placeholder, logits=self.logits5) + beta*reg)
        beta = self.l2_placeholder
        reg = 0
        for v in tf.trainable_variables():
            reg = reg + tf.nn.l2_loss(v)
        self.loss6 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.label_placeholder, logits=self.logits6) + beta*reg)
        beta = self.l2_placeholder
        reg = 0
        for v in tf.trainable_variables():
            reg = reg + tf.nn.l2_loss(v)
        self.loss9 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.label_placeholder, logits=self.logits9) + beta*reg)

    def add_train_op(self):
        #if self.length == 1256:
        if True:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss1256, tvars), self.config.max_grad_norm)
            self.train_op1256 = optimizer.apply_gradients(zip(grads, tvars))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss1, tvars), self.config.max_grad_norm)
        self.train_op1 = optimizer.apply_gradients(zip(grads, tvars))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss2, tvars), self.config.max_grad_norm)
        self.train_op2 = optimizer.apply_gradients(zip(grads, tvars))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss5, tvars), self.config.max_grad_norm)
        self.train_op5 = optimizer.apply_gradients(zip(grads, tvars))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss6, tvars), self.config.max_grad_norm)
        self.train_op6 = optimizer.apply_gradients(zip(grads, tvars))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss9, tvars), self.config.max_grad_norm)
        self.train_op9 = optimizer.apply_gradients(zip(grads, tvars))

    def optimize(self, sess, prem_batch, prem_len, hyp_batch, hyp_len, label_batch, length):
        if length != 1256:
            input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len, self.config.dropout, self.config.l2_norm, self.config.learning_rate, label_batch)
        else:
            input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len, self.config.dropout, self.config.l2_norm, self.config.learning_rate, None, label_batch)
        self.length = length
        if length == 1256:
            output_feed = [self.train_op1256, self.logits1256, self.loss1256]
            _, logits, loss = sess.run(output_feed, input_feed)
            return None, loss
        if length == 1:
            output_feed = [self.train_op1, self.logits1, self.loss1]
            _, logits, loss = sess.run(output_feed, input_feed)
            return np.argmax(logits, axis=1), loss
        if length == 2:
            output_feed = [self.train_op2, self.logits2, self.loss2]
            _, logits, loss = sess.run(output_feed, input_feed)
            return np.argmax(logits, axis=1), loss
        if length == 5:
            output_feed = [self.train_op5, self.logits5, self.loss5]
            _, logits, loss = sess.run(output_feed, input_feed)
            return np.argmax(logits, axis=1), loss
        if length == 6:
            output_feed = [self.train_op6, self.logits6, self.loss6]
            _, logits, loss = sess.run(output_feed, input_feed)
            return np.argmax(logits, axis=1), loss
        if length == 9:
            output_feed = [self.train_op9, self.logits9, self.loss9]
            _, logits, loss = sess.run(output_feed, input_feed)
            return np.argmax(logits, axis=1), loss

    def predict(self, sess, prem_batch, prem_len, hyp_batch, hyp_len, label_batch):
        input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len,1, 0, 0, label_batch)
        output_feed = [self.logits9, self.loss9]
        logits, loss = sess.run(output_feed, input_feed)
        return np.argmax(logits, axis=1), loss

    def run_train_epoch(self, sess, dataset):
        print(np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))
        preds = []
        labels = []
        losses = 0.
        x = 0
        count = 0
        for prem, prem_len, hyp, hyp_len, label in dataset:
            pred, loss = self.optimize(sess, prem, prem_len, hyp, hyp_len,label)
            preds.extend(pred)
            labels.extend(label)
            losses += loss * len(label)
            x += 1
            count +=1
        return preds, labels, losses / len(labels)

    def run_test_epoch(self, sess, dataset):
        preds = []
        labels = []
        losses = 0.
        for prem, prem_len, hyp, hyp_len, label, _ in dataset:
            pred, loss = self.predict(sess, prem, prem_len, hyp, hyp_len, label)
            preds.extend(pred)
            labels.extend(label)
            losses += loss * len(label)
        return preds, labels, losses / len(labels)
