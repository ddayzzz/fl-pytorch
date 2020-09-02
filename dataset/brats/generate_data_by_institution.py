import pandas as pd
import os
import glob
import json
import argparse

def get_data_info(prefix):
    data_info_path = os.path.join(prefix, 'data_info.csv')
    data_info = pd.read_csv(data_info_path)
    # 按照机构进行选择
    ins_to_files = dict()
    def sort_by_slice(x):
        x = os.path.basename(x).split('_')[-1]
        x = x[:x.rfind('.')]
        return int(x)

    for ins in data_info['institution']:
        items = data_info[data_info['institution'] == ins]
        file_ids = items['id']
        hgg_or_lggs = items['hgg_or_lgg']
        # 读取文件
        file_id_to_image_mask = dict()
        for hgg_or_lgg, file_id in zip(hgg_or_lggs, file_ids):
            images = glob.glob(os.path.join(prefix, 'image', hgg_or_lgg + '_' + file_id + '*.npy'))
            # 存在 slice或者分块, 所以需要排序
            images = sorted(images, key=sort_by_slice)
            #
            masks = [os.path.join(prefix, 'mask', os.path.basename(x)) for x in images]
            file_id_to_image_mask[file_id] = (images, masks)
        ins_to_files[ins] = file_id_to_image_mask

    return ins_to_files


def generate_train_config(train_dataset_prefixes, test_dataset_prefixes, save_prefix, config_name):
    cfg = os.path.join(save_prefix, config_name + '.json')

    def merge(prefixes):
        infos = [get_data_info(prefix) for prefix in prefixes]
        # 具有相同 结构的数据进行合并
        new_merged = infos[0]
        for i in range(1, len(infos)):
            for ins in infos[i]:
                # 应该不存在相同的 id
                new_merged[ins].update(infos[i][ins])

        return new_merged
    train = merge(train_dataset_prefixes)
    test = merge(test_dataset_prefixes)
    with open(cfg, 'w') as fp:
        json.dump({
            'train': train,
            'test': test
        }, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='/home/liuyuan/shu_codes/datasets/brats_patch_3d_128_128/brats2018_3d_fl', type=str)
    parser.add_argument('--test', default='/home/liuyuan/shu_codes/datasets/brats_patch_3d_128_128/brats2019_3d_fl', type=str)
    root = os.path.dirname(os.path.realpath(__file__))
    data = os.path.join(root, 'data', 'configs')
    os.makedirs(data, exist_ok=True)

    opt = parser.parse_args().__dict__
    trains = [
        opt['train']
    ]
    tests = [
        opt['test']
    ]
    generate_train_config(trains, tests, save_prefix=data, config_name='2018train_2019test')

