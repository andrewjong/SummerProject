class config:
    vocab_limit = 65536
    vocab_dim = 300
    retrain_embeddings = False
    num_classes = 3
    max_prem_len = 20 
    max_hyp_len = 20
    dropout = 0.8
    num_layers = 1
    state_size = 32
    max_grad_norm = 5.
    batch_size = 256
    l2 = 0.0001
    num_epoch = 10
    data_path = "data/trainingData"
