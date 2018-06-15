def model_opts(parser):

    # If you prefer configuration in a config file, you can put the setting in a yaml file.
    # When there is conflict between opt and config, the system prefers the setting in config. For more details, see the function "convert_to_config" at the bottom.
    parser.add_argument('-config', default='default.yaml', type=str,
                        help="config file")

    parser.add_argument('-gpus', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")

    parser.add_argument('-restore', default='', type=str,
                        help="restore checkpoint")
    parser.add_argument('-seed', type=int, default=1234,
                        help="Random seed")
    parser.add_argument('-model', default='seq2seq', type=str,
                        help="Model selection")
    parser.add_argument('-mode', default='train', type=str,
                        help="Mode selection")
    parser.add_argument('-module', default='seq2seq', type=str,
                        help="Module selection")

    parser.add_argument('-log', default='', type=str,
                        help="log directory")
    parser.add_argument('-num_processes', type=int, default=4,
                        help="number of processes")
    parser.add_argument('-refF', default='', type=str,
                        help="reference file")

    parser.add_argument('-unk', action='store_true', help='replace unk')
    parser.add_argument('-char', action='store_true', help='char level decoding')
    parser.add_argument('-length_norm', action='store_true', help='replace unk')

    # config
    parser.add_argument('-batch_size', type=int, default=64, help="batch size")
    parser.add_argument('-optim', default='adam', type=str, help="optimizer")
    parser.add_argument('-cell', default='lstm', type=str, help="cell for rnn")
    parser.add_argument('-attention', default='luong_gate', type=str, help="attention mechanism")
    parser.add_argument('-learning_rate', default=0.0003, type=float, help="learning rate")
    parser.add_argument('-max_grad_norm', type=int, default=5, help="maximum gradient norm")
    parser.add_argument('-learning_rate_decay', default=0.5, type=float, help="decay rate for learning rate")
    parser.add_argument('-start_decay_at', default=5, type=int, help="the epoch when the learning rate decays")
    parser.add_argument('-emb_size', default=512, type=int, help="embedding size")
    parser.add_argument('-hidden_size', default=512, type=int, help="hidden size")
    parser.add_argument('-enc_num_layers', default=2, type=int, help="number of layers for the encoder")
    parser.add_argument('-dec_num_layers', default=2, type=int, help="number of layers for the decoder")
    parser.add_argument('-bidirectional', action='store_true', help="bidirectional rnn for the encoder")
    parser.add_argument('-dropout', default=0.0, type=float, help="dropout rate")
    parser.add_argument('-max_time_step', default=100, type=int, help="maximum time steps for generation")
    parser.add_argument('-eval_interval', default=5000, type=int, help="runs for each evaluation")
    parser.add_argument('-save_interval', default=3000, type=int, help="runs for saving checkpoint")
    parser.add_argument('-metrics', default=[], nargs='+', type=str, help="metric for evaluation")
    parser.add_argument('-shared_vocab', action='store_true', help="shared vocabulary for the encoder and decoder")
    parser.add_argument('-beam_size', default=10, type=int, help="beam size")
    parser.add_argument('-schedule', action='store_true', help="learning rate decay schedule")
    parser.add_argument('-schesamp', action='store_true', help="schedule sampling")

    # gated
    parser.add_argument('-gate', action='store_true', help='gated')

    # global encoding
    parser.add_argument('-swish', action='store_true', help='inception for global encoding')
    parser.add_argument('-selfatt', action='store_true', help='self-attention')

    # WEAN
    parser.add_argument('-score_fn', default='', type=str, help="score_fn")

    # superAE
    parser.add_argument('-sae', action='store_true', help='super-AutoEncoder')
    parser.add_argument('-loss_reg', default='l2', type=str, help="regularized loss for the states of s2s and ae")
    parser.add_argument('-ae_weight', default=0.0, type=float, help="weight for regularized loss")

    parser.add_argument('-pool_size', type=int, default=0, help="pool size of maxout layer")
    parser.add_argument('-scale', type=float, default=1, help="proportion of the training set")
    parser.add_argument('-max_split', type=int, default=0, help="max generator time steps for memory efficiency")
    parser.add_argument('-split_num', type=int, default=0, help="split number for splitres")
    parser.add_argument('-pretrain', default='', type=str, help="load pretrain encoder")

    # tensorboard
    parser.add_argument('-tensorboard', action="store_true", help="Use tensorboardX for visualization during training.")
    parser.add_argument("-tensorboard_log_dir", type=str, default="runs", help="Log directory for Tensorboard.")


def convert_to_config(opt, config):
    opt = vars(opt)
    for key in opt:
        if key not in config:
            config[key] = opt[key]
