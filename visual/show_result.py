import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import re

# plt.rc('font', family='serif')
# plt.rc('font', serif='Times New Roman')
# plt.rcParams["mathtext.fontset"] = "stix"


def get_info(metric_json):
    with open(metric_json) as fp:
        metrics = json.load(fp)
    return metrics

def get_metrics(dataset, exp_filter_re):
    prefix = f'result/{dataset}'
    dirs = os.listdir(prefix)
    dirs = reversed(dirs)
    res = dict()
    for info in dirs:
        if exp_filter_re is not None and re.search(exp_filter_re, info) is None:
            continue
        exp_name = info[info.find('wn'):]
        res[exp_name] = get_info('{}/{}/metrics.json'.format(prefix, info))
    return res


def plot(dataset, *args, exp_filter_re=None):
    infos = get_metrics(dataset, exp_filter_re)
    for arg in args:
        f = plt.figure()
        for exp, jf in infos.items():
            num_rounds = int(jf['num_rounds']) + 1
            metric = np.asarray(jf[arg])
            plt.plot(np.arange(num_rounds), metric, label=exp, linewidth=3.0)
            # plt.plot(np.asarray(rounds1[:len(losses1)]), np.asarray(losses1), '--', linewidth=3.0, label='mu=0, E=20',
            #          color="#17becf")
            # plt.legend(loc='best')
            plt.xlabel('通信轮次', fontsize=12)
            # plt.ylabel(arg, fontsize=12)
        # plt.title(dataset)
        # plt.show()
        plt.tight_layout()
        f.savefig(f'exps/exp_{dataset}_{arg}.png')

def plot_diff_dp(dataset, exps, *args):
    prefix = f'result/{dataset}'
    for arg in args:
        f = plt.figure()
        infos = dict()
        leg_infos = dict()
        for exp in exps.keys():
            exp_name = exp[exp.find('wn'):]
            leg_infos[exp_name] = exps[exp]
            infos[exp_name] = get_info('{}/{}/metrics.json'.format(prefix, exp))
        for exp, jf in infos.items():
            num_rounds = int(jf['num_rounds']) + 1
            metric = np.asarray(jf[arg])
            plt.plot(np.arange(num_rounds), metric, label=leg_infos[exp], linewidth=3.0)
        # plt.legend(loc='best', ncol=3)
        # leg = plt.gca().get_legend()  # 或leg=ax.get_legend()
        # ltext = leg.get_texts()
        # plt.setp(ltext, fontsize=12)
        plt.xlabel('通信轮次', fontsize=12)
            # plt.ylabel(arg, fontsize=12)
        # plt.title(dataset)
        # plt.show()
        plt.tight_layout()
        f.savefig(f'exps/exp_{dataset}_{arg}.png')


def plot_diff_scheme(dataset, exps, legend, *args):
    # if legend:
    #     plt.rc('font', family='serif')
    #     plt.rc('font', serif='Times New Roman')
    #     plt.rcParams["mathtext.fontset"] = "stix"
    prefix = f'result/{dataset}'
    for arg in args:
        f = plt.figure()
        infos = dict()
        leg_infos = dict()
        for exp in exps.keys():
            # exp_name = exp[exp.find('wn'):]
            leg_infos[exp] = exps[exp]
            infos[exp] = get_info('{}/{}/metrics.json'.format(prefix, exp))
        for exp, jf in infos.items():
            num_rounds = int(jf['num_rounds']) + 1
            metric = np.asarray(jf[arg])
            plt.plot(np.arange(num_rounds), metric, label=leg_infos[exp], linewidth=3.0)
        if legend:
            plt.legend(loc='best', ncol=4)
            leg = plt.gca().get_legend()  # 或leg=ax.get_legend()
            ltext = leg.get_texts()
            plt.setp(ltext, fontsize=12)
        plt.xlabel('通信轮次', fontsize=12)
        # plt.ylabel(arg, fontsize=12)
        # plt.title(dataset)
        if legend:
            plt.show()
        plt.tight_layout()
        # f.savefig(f'exps/exp_{dataset}_{arg}.png')

import pandas as pd

def get_tb_info(fn):
    df = pd.read_csv(fn)
    return df

def plot_tb_exported(dataset, exps, legend, *args):
    # if legend:
    #     plt.rc('font', family='serif')
    #     plt.rc('font', serif='Times New Roman')
    #     plt.rcParams["mathtext.fontset"] = "stix"
    prefix = f'../fedmeta_results/tensorboard 导出/{dataset}'
    for arg in args:
        # f = plt.figure()
        infos = dict()
        leg_infos = dict()
        for exp in exps.keys():
            # exp_name = exp[exp.find('wn'):]
            leg_infos[exp] = exps[exp]
            infos[exp] = get_tb_info(os.path.join(prefix, exp, arg + '.csv'))
        for exp, jf in infos.items():
            num_rounds = jf['Step']
            metric = jf['Value']
            plt.plot(num_rounds, metric, label=leg_infos[exp], linewidth=2.0)
        if legend:
            plt.legend(loc='best', ncol=4)
            plt.legend(loc='best')
        #     leg = plt.gca().get_legend()  # 或leg=ax.get_legend()
        #     ltext = leg.get_texts()
        #     plt.setp(ltext, fontsize=12)
        plt.xlabel('通信轮次', fontsize=12)
        # plt.ylabel(arg, fontsize=12)
        # plt.title(dataset)
        if legend:
            plt.show()
        plt.tight_layout()
        # f.savefig(f'exps/exp_{dataset}_{arg}.png')


def plot_computation(dataset, exps, legend, num_rounds, client_per_round):
    # if legend:
    #     plt.rc('font', family='serif')
    #     plt.rc('font', serif='Times New Roman')
    #     plt.rcParams["mathtext.fontset"] = "stix"
    prefix = f'../fedmeta_results/tensorboard 导出/{dataset}'
    arg = 'client_computations'
    # f = plt.figure()
    infos = dict()
    leg_infos = dict()
    for exp in exps.keys():
        # exp_name = exp[exp.find('wn'):]
        leg_infos[exp] = exps[exp]
        infos[exp] = get_info(os.path.join(prefix, exp, 'metrics.json'))
    for exp, jf in infos.items():
        r = np.arange(num_rounds)
        metric = np.zeros(shape=[num_rounds])
        for cid, cid_comp in infos[exp][arg].items():
            for i, comp in zip(range(num_rounds), cid_comp):
                metric[i] += comp
        comp_mean = metric / client_per_round
        plt.plot(r[::50], comp_mean[::50], label=leg_infos[exp], linewidth=2.0)
    if legend:
        # plt.legend(loc='best', ncol=4)
        plt.legend(loc='best')
    #     leg = plt.gca().get_legend()  # 或leg=ax.get_legend()
    #     ltext = leg.get_texts()
    #     plt.setp(ltext, fontsize=12)
    plt.xlabel('通信轮次', fontsize=12)
    # plt.ylabel(arg, fontsize=12)
    # plt.title(dataset)
    if legend:
        plt.show()
    plt.tight_layout()
    # f.savefig(f'exps/exp_{dataset}_{arg}.png')


def plot_tb_exported_qmaml(dataset, exps, legend, qmaml_min_acc, qmaml_max_acc, max_rounds, *args):
    # if legend:
    #     plt.rc('font', family='serif')
    #     plt.rc('font', serif='Times New Roman')
    #     plt.rcParams["mathtext.fontset"] = "stix"
    prefix = f'../fedmeta_results/tensorboard 导出/{dataset}'
    for arg in args:
        # f = plt.figure()
        infos = dict()
        leg_infos = dict()
        for exp in exps.keys():
            # exp_name = exp[exp.find('wn'):]
            leg_infos[exp] = exps[exp]
            infos[exp] = get_tb_info(os.path.join(prefix, exp, arg + '.csv'))
        for exp, jf in infos.items():
            num_rounds = jf['Step']
            metric = jf['Value']

            steps = []
            me = []
            for step, met in zip(num_rounds, metric):
                if step <= max_rounds:
                    steps.append(step)
                    me.append(met)
                else:
                    break
            plt.plot(steps, me, label=leg_infos[exp], linewidth=2.0)

        plt.plot(np.arange(0, max_rounds, 50), np.ones((max_rounds, ))[::50] * qmaml_min_acc, label='q-MAML最优 - 不开启精调', linewidth=2.0, marker='x')
        plt.plot(np.arange(0, max_rounds, 50), np.ones((max_rounds,))[::50] * qmaml_max_acc, label='q-MAML最优 - 开启精调(5个mini-batch)', linewidth=2.0, marker='o')
        if legend:
            plt.legend(loc='best')
        #     leg = plt.gca().get_legend()  # 或leg=ax.get_legend()
        #     ltext = leg.get_texts()
        #     plt.setp(ltext, fontsize=12)
        plt.xlabel('通信轮次', fontsize=12)
        # plt.ylabel(arg, fontsize=12)
        # plt.title(dataset)
        if legend:
            plt.show()
        plt.tight_layout()

        # f.savefig(f'exps/exp_{dataset}_{arg}.png')

if __name__ == '__main__':
    # fedavg scheme mnist
    dataset = 'femnist'
    # plot_tb_exported(dataset, {
    #     'fedavg 不使用所有数据': 'FedAvg(方式1)',
    #     'fedavg 使用所有数据': 'FedAvg(方式2)',
    #     'fedmeta': 'FedMeta(MAML)',
    #     'fedmeta_inner5': 'FedMeta(MAML) - 5个 mini-batch'
    #
    # }, True, 'test_acc', 'test_loss')

    # plot_computation(dataset, {
    #     'fedavg 不使用所有数据': 'FedAvg(方式1)',
    #     'fedavg 使用所有数据': 'FedAvg(方式2)',
    #     'fedmeta': 'FedMeta(MAML)',
    #     'fedmeta_inner5': 'FedMeta(MAML) - 5个 mini-batch'
    #
    # }, True, 2000, 4)

    plot_tb_exported_qmaml('omniglot', {
        'fedmeta_inner1': 'FedMeta(MAML) - 1个 mini-batch',
        'fedmeta_inner5': 'FedMeta(MAML) - 5个 mini-batch'

    }, True, 0.7448, 0.8682, 2000, 'test_acc')