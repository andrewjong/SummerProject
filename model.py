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
        self.learning_rate_placeholder = tf.placeholder(tf.float32, shape=())

    def create_feed_dict(self, prem_batch, prem_len, hyp_batch, hyp_len, dropout, l2 = None, learning_rate = None, label_batch=None):
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
            prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
            new_prems, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems,\
                          sequence_length=self.prem_len_placeholder, dtype=tf.float32)
        with tf.variable_scope("hyp"):
            hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
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
            Wy = tf.Variable(initer([1,1,self.config.state_size, self.config.state_size]))
            Wh = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            Wr = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            w =  tf.Variable(initer([1,1,self.config.state_size]))
            M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(new_hyps[:,0,:], Wh), 1))
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
            r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
            Wt = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            for i in range(1,10):
                M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(new_hyps[:,i,:], Wh), 1) + tf.expand_dims(tf.matmul(r, Wr), 1))
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
                r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1) + tf.tanh(tf.matmul(r, Wt))
            Wp = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            Wx= tf.Variable(initer([self.config.state_size, self.config.state_size]))
            h = tf.tanh(tf.matmul(r, Wp) + tf.matmul(hyp_out, Wx))
        Ws1 = tf.Variable(initer([self.config.state_size,self.config.state_size]))
        bs1 = tf.Variable(tf.zeros([1,self.config.state_size]) + 1e-3)
        h = tf.tanh(tf.matmul(h, Ws1) + bs1)
        Ws2 = tf.Variable(initer([self.config.state_size,3]))
        bs2 = tf.Variable(tf.zeros([1,3]) + 1e-3)
        self.logits = tf.matmul(h, Ws2) + bs2

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


    def combine(self,stuff, name, reuse=True, size=None):
        if size is None:
            size = self.config.state_size
        xavier = tf.contrib.layers.xavier_initializer()
        return tf.layers.dense(
                                tf.concat(stuff, 1),
                                size,
                                activation=tf.nn.relu,
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
                prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                _, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems,\
                              sequence_length=self.prem_len_placeholder, dtype=tf.float32)
            with tf.variable_scope("hyp-siamese"):
                hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.config.state_size), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                _, hyp_out = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps,\
                             sequence_length=self.hyp_len_placeholder, dtype=tf.float32)

            representation = tf.layers.dense(
                                            tf.concat([prem_out.h, hyp_out.h], 1),
                                            self.config.state_size,
                                            activation=tf.nn.relu,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            )
            representation2 = tf.layers.dense(
                                            representation,
                                            self.config.state_size,
                                            activation=tf.nn.relu,
                                            kernel_initializer=xavier,
                                            use_bias=True,
                                            )

            self.logits = tf.layers.dense(representation2, 3,
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
        if self.model_type == "restcomp":
            subjectd = self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,300]), tf.reshape(self.embed_hyps[:,0,:], [-1,300])],"ycomp", reuse=False, size=16)
            subjectn = self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,300]), tf.reshape(self.embed_hyps[:,1,:], [-1,300])],"zcomp",reuse=False, size=2)
            subjecta = self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,300]), tf.reshape(self.embed_hyps[:,2,:], [-1,300])],"xcomp", reuse=False,size=4)
            neg = self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,300]), tf.reshape(self.embed_hyps[:,4,:], [-1,300])],"vcomp",reuse=False,size=4)
            verb = self.combine([tf.reshape(self.embed_prems[:,5,:], [-1,300]), tf.reshape(self.embed_hyps[:,5,:], [-1,300])],"zcomp",size=2)
            adverb = self.combine([tf.reshape(self.embed_prems[:,6,:], [-1,300]), tf.reshape(self.embed_hyps[:,6,:], [-1,300])],"xcomp",size=4)
            objectd = self.combine([tf.reshape(self.embed_prems[:,7,:], [-1,300]), tf.reshape(self.embed_hyps[:,7,:], [-1,300])],"ycomp",size=16)
            objectn = self.combine([tf.reshape(self.embed_prems[:,8,:], [-1,300]), tf.reshape(self.embed_hyps[:,8,:], [-1,300])],"zcomp",size=2)
            objecta = self.combine([tf.reshape(self.embed_prems[:,9,:], [-1,300]), tf.reshape(self.embed_hyps[:,9,:], [-1,300])],"xcomp",size=4)
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

        if self.model_type == "simpcomp":
            subjectd = self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,300]), tf.reshape(self.embed_hyps[:,0,:], [-1,300])],"comp", reuse=False)
            subjectn = self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,300]), tf.reshape(self.embed_hyps[:,1,:], [-1,300])],"comp")
            subjecta = self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,300]), tf.reshape(self.embed_hyps[:,2,:], [-1,300])],"comp")
            neg = self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,300]), tf.reshape(self.embed_hyps[:,4,:], [-1,300])],"comp")
            verb = self.combine([tf.reshape(self.embed_prems[:,5,:], [-1,300]), tf.reshape(self.embed_hyps[:,5,:], [-1,300])],"comp")
            adverb = self.combine([tf.reshape(self.embed_prems[:,6,:], [-1,300]), tf.reshape(self.embed_hyps[:,6,:], [-1,300])],"comp")
            objectd = self.combine([tf.reshape(self.embed_prems[:,7,:], [-1,300]), tf.reshape(self.embed_hyps[:,7,:], [-1,300])],"comp")
            objectn = self.combine([tf.reshape(self.embed_prems[:,8,:], [-1,300]), tf.reshape(self.embed_hyps[:,8,:], [-1,300])],"comp")
            objecta = self.combine([tf.reshape(self.embed_prems[:,9,:], [-1,300]), tf.reshape(self.embed_hyps[:,9,:], [-1,300])],"comp")
            subjectNP = self.combine([subjecta, subjectn],"comp")
            objectNP = self.combine([objecta, objectn],"comp")
            VP = self.combine([adverb, verb],"comp")
            objectDP1 = self.combine([objectd, objectNP],"comp")
            objectDP2 = self.combine([objectDP1, VP],"comp")
            negobjectDP = self.combine([neg, objectDP2],"comp")
            final = self.combine([subjectd, subjectNP,],"comp")
            final2 = self.combine([final, negobjectDP],"comp")
            self.logits = tf.layers.dense(final2, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)

        if self.model_type == "boolcomp":
            subjectd = self.combine([tf.reshape(self.embed_prems[:,0,:], [-1,300]), tf.reshape(self.embed_hyps[:,0,:], [-1,300])],"comp", reuse=False)
            subjectn = self.combine([tf.reshape(self.embed_prems[:,1,:], [-1,300]), tf.reshape(self.embed_hyps[:,1,:], [-1,300])],"comp")
            subjecta = self.combine([tf.reshape(self.embed_prems[:,2,:], [-1,300]), tf.reshape(self.embed_hyps[:,2,:], [-1,300])],"comp")
            neg = self.combine([tf.reshape(self.embed_prems[:,4,:], [-1,300]), tf.reshape(self.embed_hyps[:,4,:], [-1,300])],"comp")
            verb = self.combine([tf.reshape(self.embed_prems[:,5,:], [-1,300]), tf.reshape(self.embed_hyps[:,5,:], [-1,300])],"comp")
            adverb = self.combine([tf.reshape(self.embed_prems[:,6,:], [-1,300]), tf.reshape(self.embed_hyps[:,6,:], [-1,300])],"comp")
            objectd = self.combine([tf.reshape(self.embed_prems[:,7,:], [-1,300]), tf.reshape(self.embed_hyps[:,7,:], [-1,300])],"comp")
            objectn = self.combine([tf.reshape(self.embed_prems[:,8,:], [-1,300]), tf.reshape(self.embed_hyps[:,8,:], [-1,300])],"comp")
            objecta = self.combine([tf.reshape(self.embed_prems[:,9,:], [-1,300]), tf.reshape(self.embed_hyps[:,9,:], [-1,300])],"comp")
            subjectNP = self.combine([subjecta, subjectn],"comp")
            objectNP = self.combine([objecta, objectn],"comp")
            VP = self.combine([adverb, verb],"comp")
            objectDP1 = self.combine([objectd, objectNP],"comp")
            objectDP2 = self.combine([objectDP1, VP],"comp")
            negobjectDP = self.combine([neg, objectDP2],"comp")
            final = self.combine([subjectd, subjectNP,],"comp")
            final2 = self.combine([final, negobjectDP],"comp")
            subjectd2 = self.combine([tf.reshape(self.embed_prems[:,0 + 11,:], [-1,300]), tf.reshape(self.embed_hyps[:,0 + 11,:], [-1,300])],"comp")
            subjectn2 = self.combine([tf.reshape(self.embed_prems[:,1 + 11,:], [-1,300]), tf.reshape(self.embed_hyps[:,1 + 11,:], [-1,300])],"comp")
            subjecta2 = self.combine([tf.reshape(self.embed_prems[:,2 + 11,:], [-1,300]), tf.reshape(self.embed_hyps[:,2 + 11,:], [-1,300])],"comp")
            neg2 = self.combine([tf.reshape(self.embed_prems[:,4 + 11,:], [-1,300]), tf.reshape(self.embed_hyps[:,4 + 11,:], [-1,300])],"comp")
            verb2 = self.combine([tf.reshape(self.embed_prems[:,5 + 11,:], [-1,300]), tf.reshape(self.embed_hyps[:,5 + 11,:], [-1,300])],"comp")
            adverb2 = self.combine([tf.reshape(self.embed_prems[:,6 + 11,:], [-1,300]), tf.reshape(self.embed_hyps[:,6 + 11,:], [-1,300])],"comp")
            objectd2 = self.combine([tf.reshape(self.embed_prems[:,7 + 11,:], [-1,300]), tf.reshape(self.embed_hyps[:,7 + 11,:], [-1,300])],"comp")
            objectn2 = self.combine([tf.reshape(self.embed_prems[:,8 + 11,:], [-1,300]), tf.reshape(self.embed_hyps[:,8 + 11,:], [-1,300])],"comp")
            objecta2 = self.combine([tf.reshape(self.embed_prems[:,9 + 11,:], [-1,300]), tf.reshape(self.embed_hyps[:,9 + 11,:], [-1,300])],"comp")
            subjectNP2 = self.combine([subjecta2, subjectn2],"comp")
            objectNP2 = self.combine([objecta2, objectn2],"comp")
            VP2 = self.combine([adverb2, verb2],"comp")
            objectDP12 = self.combine([objectd2, objectNP2],"comp")
            objectDP22 = self.combine([objectDP12, VP2],"comp")
            negobjectDP2 = self.combine([neg2, objectDP22],"comp")
            final2 = self.combine([subjectd2, subjectNP2],"comp")
            final22 = self.combine([final2, negobjectDP2],"comp")
            #final2 = tf.nn.softmax(tf.layers.dense(final2, 3,
            #                              kernel_initializer=xavier,
            #                              use_bias=True))
            #final22 = tf.nn.softmax(tf.layers.dense(final22, 3,
            #                              kernel_initializer=xavier,
            #                              use_bias=True))
            conj = self.combine([tf.reshape(self.embed_prems[:,10,:], [-1,300]), tf.reshape(self.embed_hyps[:,10,:], [-1,300])],"comp")
            truefinal1 = self.combine([conj, final2], "comp")
            truefinal2 = self.combine([truefinal1,final22], "comp")
            self.logits = tf.layers.dense(truefinal2, 3,
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
            psubjectd = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,0,:], [-1,300]))
            psubjectn = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,1,:], [-1,300]))
            psubjecta = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,2,:], [-1,300]))
            pneg = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,4,:], [-1,300]))
            pverb = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,5,:], [-1,300]))
            padverb = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,6,:], [-1,300]))
            pobjectd = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,7,:], [-1,300]))
            pobjectn = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,8,:], [-1,300]))
            pobjecta = self.LSTMcombine(input= tf.reshape(self.embed_prems[:,9,:], [-1,300]))
            psubjectNP = self.LSTMcombine(children=[psubjecta, psubjectn])
            pobjectNP = self.LSTMcombine(children=[pobjecta, pobjectn])
            pVP = self.LSTMcombine(children=[padverb, pverb])
            pobjectDP1 = self.LSTMcombine(children=[pobjectd, pobjectNP])
            pobjectDP2 = self.LSTMcombine(children=[pobjectDP1, pVP])
            pnegobjectDP = self.LSTMcombine(children=[pneg, pobjectDP2])
            pfinal = self.LSTMcombine(children=[psubjectd, psubjectNP,])
            pfinal2 = self.LSTMcombine(children=[pfinal, pnegobjectDP])
            hsubjectd = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,0,:], [-1,300]))
            hsubjectn = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,1,:], [-1,300]))
            hsubjecta = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,2,:], [-1,300]))
            hneg = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,4,:], [-1,300]))
            hverb = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,5,:], [-1,300]))
            hadverb = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,6,:], [-1,300]))
            hobjectd = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,7,:], [-1,300]))
            hobjectn = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,8,:], [-1,300]))
            hobjecta = self.LSTMcombine(input= tf.reshape(self.embed_hyps[:,9,:], [-1,300]))
            hsubjectNP = self.LSTMcombine(children=[hsubjecta, hsubjectn])
            hobjectNP = self.LSTMcombine(children=[hobjecta, hobjectn])
            hVP = self.LSTMcombine(children=[hadverb, hverb])
            hobjectDP1 = self.LSTMcombine(children=[hobjectd, hobjectNP])
            hobjectDP2 = self.LSTMcombine(children=[hobjectDP1, hVP])
            hnegobjectDP = self.LSTMcombine(children=[hneg, hobjectDP2])
            hfinal = self.LSTMcombine(children=[hsubjectd, hsubjectNP,])
            hfinal2 = self.LSTMcombine(children=[hfinal, hnegobjectDP])
            final = self.combine([pfinal2[0], hfinal2[0]], "final", reuse=False)
            final2 = self.combine([final], "final2", reuse=False)
            self.logits = tf.layers.dense(final2, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)

        if self.model_type == "sepsimpcomp":
            psubjectd = tf.reshape(self.embed_prems[:,0,:], [-1,300])
            psubjectn = tf.reshape(self.embed_prems[:,1,:], [-1,300])
            psubjecta = tf.reshape(self.embed_prems[:,2,:], [-1,300])
            pneg = tf.reshape(self.embed_prems[:,4,:], [-1,300])
            pverb = tf.reshape(self.embed_prems[:,5,:], [-1,300])
            padverb = tf.reshape(self.embed_prems[:,6,:], [-1,300])
            pobjectd = tf.reshape(self.embed_prems[:,7,:], [-1,300])
            pobjectn = tf.reshape(self.embed_prems[:,8,:], [-1,300])
            pobjecta = tf.reshape(self.embed_prems[:,9,:], [-1,300])
            psubjectNP = self.combine([psubjecta, psubjectn],"comp",reuse=False)
            pobjectNP = self.combine([pobjecta, pobjectn],"comp")
            pVP = self.combine([padverb, pverb],"comp")
            pobjectDP1 = self.combine([pobjectd, pobjectNP],"comp")
            pobjectDP2 = self.combine([pobjectDP1, pVP],"comp")
            pnegobjectDP = self.combine([pneg, pobjectDP2],"comp")
            pfinal = self.combine([psubjectd, psubjectNP,],"comp")
            pfinal2 = self.combine([pfinal, pnegobjectDP],"comp")
            hsubjectd = tf.reshape(self.embed_hyps[:,0,:], [-1,300])
            hsubjectn = tf.reshape(self.embed_hyps[:,1,:], [-1,300])
            hsubjecta = tf.reshape(self.embed_hyps[:,2,:], [-1,300])
            hneg = tf.reshape(self.embed_hyps[:,4,:], [-1,300])
            hverb = tf.reshape(self.embed_hyps[:,5,:], [-1,300])
            hadverb = tf.reshape(self.embed_hyps[:,6,:], [-1,300])
            hobjectd = tf.reshape(self.embed_hyps[:,7,:], [-1,300])
            hobjectn = tf.reshape(self.embed_hyps[:,8,:], [-1,300])
            hobjecta = tf.reshape(self.embed_hyps[:,9,:], [-1,300])
            hsubjectNP = self.combine([hsubjecta, hsubjectn],"comp2", reuse=False)
            hobjectNP = self.combine([hobjecta, hobjectn],"comp2")
            hVP = self.combine([hadverb, hverb],"comp2")
            hobjectDP1 = self.combine([hobjectd, hobjectNP],"comp2")
            hobjectDP2 = self.combine([hobjectDP1, hVP],"comp2")
            hnegobjectDP = self.combine([hneg, hobjectDP2],"comp2")
            hfinal = self.combine([hsubjectd, hsubjectNP,],"comp2")
            hfinal2 = self.combine([hfinal, hnegobjectDP],"comp2")
            final = self.combine([pfinal2, hfinal2], "final", reuse=False)
            final2 = self.combine([final], "final2", reuse=False)
            premise_nodes = [psubjectd,psubjecta,psubjectn,pneg, padverb, pverb, pobjectd,pobjecta,pobjectn,psubjectNP, pobjectNP, pVP, pobjectDP1, pobjectDP2,pnegobjectDP, pfinal, pfinal2]
            premise_nodes = [tf.expand_dims(x,1) for x in premise_nodes]
            premise_nodes = tf.concat(premise_nodes,1)
            hypothesis_nodes = [hsubjectd,hsubjecta,hsubjectn,hneg, hadverb, hverb, hobjectd,hobjecta,hobjectn,hsubjectNP, hobjectNP, hVP, hobjectDP1, hobjectDP2,hnegobjectDP, hfinal, hfinal2]
            hypothesis_nodes = [tf.expand_dims(x,1) for x in hypothesis_nodes]
            hypothesis_nodes = tf.concat(hypothesis_nodes,1)
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
                final2 = tf.tanh(tf.matmul(r, Wp) + tf.matmul(final2, Wx))
            self.logits = tf.layers.dense(final2, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)

        if self.model_type == "sepboolcomp":
            psubjectd = tf.reshape(self.embed_prems[:,0,:], [-1,300])
            psubjectn = tf.reshape(self.embed_prems[:,1,:], [-1,300])
            psubjecta = tf.reshape(self.embed_prems[:,2,:], [-1,300])
            pneg = tf.reshape(self.embed_prems[:,4,:], [-1,300])
            pverb = tf.reshape(self.embed_prems[:,5,:], [-1,300])
            padverb = tf.reshape(self.embed_prems[:,6,:], [-1,300])
            pobjectd = tf.reshape(self.embed_prems[:,7,:], [-1,300])
            pobjectn = tf.reshape(self.embed_prems[:,8,:], [-1,300])
            pobjecta = tf.reshape(self.embed_prems[:,9,:], [-1,300])
            psubjectNP = self.combine([psubjecta, psubjectn],"comp",reuse=False)
            pobjectNP = self.combine([pobjecta, pobjectn],"comp")
            pVP = self.combine([padverb, pverb],"comp")
            pobjectDP1 = self.combine([pobjectd, pobjectNP],"comp")
            pobjectDP2 = self.combine([pobjectDP1, pVP],"comp")
            pnegobjectDP = self.combine([pneg, pobjectDP2],"comp")
            pfinal = self.combine([psubjectd, psubjectNP,],"comp")
            pfinal2 = self.combine([pfinal, pnegobjectDP],"comp")
            psubjectd2 = tf.reshape(self.embed_prems[:,0+11,:], [-1,300])
            psubjectn2 = tf.reshape(self.embed_prems[:,1+11,:], [-1,300])
            psubjecta2 = tf.reshape(self.embed_prems[:,2+11,:], [-1,300])
            pneg2 = tf.reshape(self.embed_prems[:,4+11,:], [-1,300])
            pverb2 = tf.reshape(self.embed_prems[:,5+11,:], [-1,300])
            padverb2 = tf.reshape(self.embed_prems[:,6+11,:], [-1,300])
            pobjectd2 = tf.reshape(self.embed_prems[:,7+11,:], [-1,300])
            pobjectn2 = tf.reshape(self.embed_prems[:,8+11,:], [-1,300])
            pobjecta2 = tf.reshape(self.embed_prems[:,9+11,:], [-1,300])
            psubjectNP2 = self.combine([psubjecta2, psubjectn2],"comp")
            pobjectNP2 = self.combine([pobjecta2, pobjectn2],"comp")
            pVP2 = self.combine([padverb2, pverb2],"comp")
            pobjectDP12 = self.combine([pobjectd2, pobjectNP2],"comp")
            pobjectDP22 = self.combine([pobjectDP12, pVP2],"comp")
            pnegobjectDP2 = self.combine([pneg2, pobjectDP2],"comp")
            pfinal2 = self.combine([psubjectd2, psubjectNP,],"comp")
            pfinal22 = self.combine([pfinal2, pnegobjectDP2],"comp")
            pconj = self.combine([tf.reshape(self.embed_prems[:,10,:], [-1,300]), tf.reshape(self.embed_hyps[:,10,:], [-1,300])],"comp")
            ptruefinal1 = self.combine([pconj, pfinal2], "comp")
            ptruefinal2 = self.combine([ptruefinal1,pfinal22], "comp")
            hsubjectd = tf.reshape(self.embed_hyps[:,0,:], [-1,300])
            hsubjectn = tf.reshape(self.embed_hyps[:,1,:], [-1,300])
            hsubjecta = tf.reshape(self.embed_hyps[:,2,:], [-1,300])
            hneg = tf.reshape(self.embed_hyps[:,4,:], [-1,300])
            hverb = tf.reshape(self.embed_hyps[:,5,:], [-1,300])
            hadverb = tf.reshape(self.embed_hyps[:,6,:], [-1,300])
            hobjectd = tf.reshape(self.embed_hyps[:,7,:], [-1,300])
            hobjectn = tf.reshape(self.embed_hyps[:,8,:], [-1,300])
            hobjecta = tf.reshape(self.embed_hyps[:,9,:], [-1,300])
            hsubjectNP = self.combine([hsubjecta, hsubjectn],"comp2", reuse=False)
            hobjectNP = self.combine([hobjecta, hobjectn],"comp2")
            hVP = self.combine([hadverb, hverb],"comp2")
            hobjectDP1 = self.combine([hobjectd, hobjectNP],"comp2")
            hobjectDP2 = self.combine([hobjectDP1, hVP],"comp2")
            hnegobjectDP = self.combine([hneg, hobjectDP2],"comp2")
            hfinal = self.combine([hsubjectd, hsubjectNP,],"comp2")
            hfinal2 = self.combine([hfinal, hnegobjectDP],"comp2")
            hsubjectd2 = tf.reshape(self.embed_hyps[:,0+11,:], [-1,300])
            hsubjectn2 = tf.reshape(self.embed_hyps[:,1+11,:], [-1,300])
            hsubjecta2 = tf.reshape(self.embed_hyps[:,2+11,:], [-1,300])
            hneg2 = tf.reshape(self.embed_hyps[:,4+11,:], [-1,300])
            hverb2 = tf.reshape(self.embed_hyps[:,5+11,:], [-1,300])
            hadverb2 = tf.reshape(self.embed_hyps[:,6+11,:], [-1,300])
            hobjectd2 = tf.reshape(self.embed_hyps[:,7+11,:], [-1,300])
            hobjectn2 = tf.reshape(self.embed_hyps[:,8+11,:], [-1,300])
            hobjecta2 = tf.reshape(self.embed_hyps[:,9+11,:], [-1,300])
            hsubjectNP2 = self.combine([hsubjecta2, hsubjectn2],"comp2")
            hobjectNP2 = self.combine([hobjecta2, hobjectn2],"comp2")
            hVP2 = self.combine([hadverb2, hverb2],"comp2")
            hobjectDP12 = self.combine([hobjectd2, hobjectNP2],"comp2")
            hobjectDP22 = self.combine([hobjectDP12, hVP2],"comp2")
            hnegobjectDP2 = self.combine([hneg2, hobjectDP22],"comp2")
            hfinal2 = self.combine([hsubjectd2, hsubjectNP2,],"comp2")
            hfinal22 = self.combine([hfinal2, hnegobjectDP2],"comp2")
            hconj = self.combine([tf.reshape(self.embed_prems[:,10,:], [-1,300]), tf.reshape(self.embed_hyps[:,10,:], [-1,300])],"comp")
            htruefinal1 = self.combine([hconj, hfinal2], "comp2")
            htruefinal2 = self.combine([htruefinal1,hfinal22], "comp2")
            final = self.combine([ptruefinal2, htruefinal2], "final", reuse=False)
            final2 = self.combine([final], "final2", reuse=False)
            self.logits = tf.layers.dense(final2, 3,
                                          kernel_initializer=xavier,
                                          use_bias=True)




    def add_loss_op(self):
        beta = self.l2_placeholder
        reg = 0
        for v in tf.trainable_variables():
            reg = reg + tf.nn.l2_loss(v)
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.label_placeholder, logits=self.logits) + beta*reg)

    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def optimize(self, sess, prem_batch, prem_len, hyp_batch, hyp_len, dropout,label_batch, lr, l2):
        input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len, dropout, l2, lr, label_batch)
        output_feed = [self.train_op, self.logits, self.loss]
        _, logits, loss = sess.run(output_feed, input_feed)
        return np.argmax(logits, axis=1), loss

    def predict(self, sess, prem_batch, prem_len, hyp_batch, hyp_len, label_batch):
        input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len,1, 0, 0, label_batch)
        output_feed = [self.logits, self.loss]
        logits, loss = sess.run(output_feed, input_feed)
        return np.argmax(logits, axis=1), loss

    def run_train_epoch(self, sess, dataset, lr, dropout, l2):
        print(np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))
        preds = []
        labels = []
        losses = 0.
        x = 0
        count = 0
        for prem, prem_len, hyp, hyp_len, label in dataset:
            pred, loss = self.optimize(sess, prem, prem_len, hyp, hyp_len, dropout, label, lr, l2)
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
        for prem, prem_len, hyp, hyp_len, label in dataset:
            pred, loss = self.predict(sess, prem, prem_len, hyp, hyp_len, label)
            preds.extend(pred)
            labels.extend(label)
            losses += loss * len(label)
        return preds, labels, losses / len(labels)
