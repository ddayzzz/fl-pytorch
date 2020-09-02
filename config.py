# GLOBAL PARAMETERS
import argparse
import warnings

DATASETS = ['shakespeare', 'mnist', 'sent140', 'cifar100', 'emnist', 'brats']
TRAINERS = {
    'fedavg': 'FedAvg',
    'fedprox': 'FedProx',
    'fedfuse': 'FedFuse',
    'fedavg_tff': 'FedAvgTFF',
    'adaptive': 'AdaptiveOptimization',
    'adaptive_medical': 'AdaptiveOptimization'
}

TRAINER_NAMES = TRAINERS.keys()
MODEL_CONFIG = {
    'mnist.logistic': {'out_dim': 10, 'in_dim': 784},
    'mnist.cnn_att':  {'image_size': 28},
    'mnist.cnn': {'image_size': 28},
    'emnist.cnn': {'num_classes': 10, 'image_size': 28},
    'emnist.cnn_cr': {'num_classes': 62, },
    'omniglot.cnn': {'num_classes': 5, 'image_size': 28},
    'shakespeare.stacked_lstm': {'seq_len': 80, 'num_classes': 80, 'num_hidden': 256, },
    'sent140.stacked_lstm': {'seq_len': 25, 'num_classes': 2, 'n_hidden': 100, 'embedding_dim': 300},
    'cifar100.resnet18_gn': {},
    'cifar100.resnet56': {},
    'brats.unet3d': {'init_channels': 4, 'class_nums': 3, 'batch_norm': True},
}




def base_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=TRAINER_NAMES,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        required=True)
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='logistic')

    parser.add_argument('--device',
                        help='device',
                        default='cpu:0',
                        type=str)
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=200)
    parser.add_argument('--eval_on_test_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--eval_on_train_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=100)
    parser.add_argument('--eval_on_validation_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--save_every',
                        help='save global model every ____ rounds;',
                        type=int,
                        default=100)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--train_batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=32)
    parser.add_argument('--test_batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=100)
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=20)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--quiet',
                        help='仅仅显示结果的代码',
                        type=int,
                        default=0)
    parser.add_argument('--result_prefix',
                        help='保存结果的前缀路径',
                        type=str,
                        default='./result')
    parser.add_argument('--num_workers', type=int, default=0)
    # TODO 以后支持 之家加载 leaf 目录里的数据
    parser.add_argument('--data_prefix', type=str, default='./')
    parser.add_argument('--data_format', type=str, default='', choices=['', 'leaf', 'pytorch'])
    return parser


def only_client_optimizer(parser):
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.01)
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--momentum', type=float, default=0.5)
    return parser


def add_dynamic_options(parser):
    # 获取对应的 solver 的名称
    params = parser.parse_known_args()[0]
    algo = params.algo
    dataset = params.dataset
    # for example
    if algo in ['fedprox']:
        parser.add_argument('--mu', help='mu', type=float, default=0.1)
        parser.add_argument('--drop_rate', help='drop rate', default=0.0, type=float)
        only_client_optimizer(parser)
    elif algo == 'fedfuse':
        parser.add_argument('--operator', help='fuse operator', type=str, required=True, choices=['multi', 'conv', 'single'])
        only_client_optimizer(parser)
    elif algo == 'fedavg_tff':
        parser.add_argument('--client_optimizer', help='learning rate for each client', default='sgd', type=str, choices=['sgd'])
        parser.add_argument('--client_lr', help='learning rate for each client', default=0.1, type=float)
        parser.add_argument('--server_lr', help='learning rate for server', default=1.0, type=float)
        parser.add_argument('--server_optimizer', help='optimizer for server', default='sgd', type=str, choices=['sgd'])
    elif algo.startswith('adaptive'):
        parser.add_argument('--client_optimizer', help='learning rate for each client', default='sgd', type=str,
                            choices=['sgd'])
        parser.add_argument('--client_lr', help='learning rate for each client', default=0.1, type=float)
        parser.add_argument('--server_lr', help='learning rate for server', default=1.0, type=float)
        parser.add_argument('--server_optimizer', help='optimizer for server', default='sgd', type=str, choices=['sgd', 'adam'])
        parser.add_argument('--wd', help='weight decay', default=0.0, type=float)
        # 以下的参数, 不同的优化器要求不同!
        parser.add_argument('--adaptive_epsilon', help='epsilon for adam-like optimizer', default=1e-7, type=float)
        parser.add_argument('--adaptive_momentum', help='learning rate for server', default=0.9, type=float, choices=[0.9, 0.0])
        parser.add_argument('--lr_decay_policy', default='constant', type=str, choices=['constant', 'inv_sqrt', 'inv_lin', 'exp_decay'])
        parser.add_argument('--lr_decay_rate', default=None, type=float)
        parser.add_argument('--lr_decay_steps', default=None, type=int)
        parser.add_argument('--lr_staircase', default=False, action='store_true')
        parser.add_argument('--lr_warmup_steps', default=None, type=int)
    else:
        only_client_optimizer(parser)

    # 添加数据相关的参数
    if dataset.startswith('cifar100'):
        parser.add_argument('--cifar100_image_size', help='crop image size', type=int, default=32)
    elif dataset.startswith('brats'):
        parser.add_argument('--brats_config', help='crop image size', type=str, default='2018train_2019test')
    return parser
