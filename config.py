# GLOBAL PARAMETERS
import argparse

DATASETS = ['shakespeare', 'mnist', 'sent140']
TRAINERS = {
    'fedavg': 'FedAvg',
    'fedprox': 'FedProx',
    'fedfuse': 'FedFuse'
}

TRAINER_NAMES = TRAINERS.keys()
MODEL_CONFIG = {
    'mnist.logistic': {'out_dim': 10, 'in_dim': 784},
    'mnist.cnn_att':  {'image_size': 28},
    'mnist.cnn': {'image_size': 28},
    'femnist.cnn': {'num_classes': 62, 'image_size': 28},
    'omniglot.cnn': {'num_classes': 5, 'image_size': 28},
    'shakespeare.stacked_lstm': {'seq_len': 80, 'num_classes': 80, 'num_hidden': 256, },
    'sent140.stacked_lstm': {'seq_len': 25, 'num_classes': 2, 'n_hidden': 100, 'embedding_dim': 300},
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
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--momentum', type=float, default=0.5)
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
                        default=50)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=20)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.01)
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
    parser.add_argument('--leaf', action='store_true', default=False)
    return parser


def add_dynamic_options(parser):
    # 获取对应的 solver 的名称
    params = parser.parse_known_args()[0]
    algo = params.algo
    # for example
    if algo in ['fedprox']:
        parser.add_argument('--mu', help='mu', type=float, default=0.1)
        parser.add_argument('--drop_rate', help='drop rate', default=0.0, type=float)
    elif algo == 'fedfuse':
        parser.add_argument('--operator', help='fuse operator', type=str, required=True, choices=['multi', 'conv', 'single'])
    return parser
