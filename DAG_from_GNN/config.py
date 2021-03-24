"""
Contains config parameters for app
"""


class CONFIG:
    """Dataclass with app parameters"""

    def __init__(self):
        pass

    # You must change this to the filename you wish to use as input data!
    data_filename = 'alarm.csv'


    data_type = 'discrete'

    data_sample_size = 5000
    graph_type = 'erdos-renyi'
    graph_degree = 2
    graph_sem_type = 'linear-gauss'
    graph_linear_type = 'nonlinear_2'
    edge_types = 2
    x_dims = 1
    z_dims = 1
    optimizer = 'Adam'
    graph_threshold = 0.3
    tau_A = 0.0
    lambda_A = 0.0
    c_A = 1
    use_A_connect_loss = 0
    use_A_positiver_loss = 0
    no_cuda = True
    seed = 42
    epochs = 300
    batch_size = 50 # note: should be divisible by sample size, otherwise throw an error
    lr = 1e-3  # basline rate = 1e-3
    encoder_hidden = 64
    decoder_hidden = 64
    temp = 0.5
    k_max_iter = 1e2
    encoder ='mlp'
    decoder = 'mlp'
    no_factor = False
    suffix = '_springs5'
    encoder_dropout = 0.0
    decoder_dropout = 0.0,
    h_tol = 1e-8
    prediction_steps = 10
    lr_decay = 200
    gamma = 1.0
    skip_first = False
    var = 5e-5
    hard = False
    prior = False
    dynamic_graph = False
