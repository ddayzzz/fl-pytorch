from torch.utils.data import Dataset
import pandas as pd


def print_stats(data, users, title):
    """
    统计信息, 必须满足需要序列化的数据的格式
    :param data:
    :param max_class_num:
    :return:
    """
    df = pd.DataFrame({'samples_per_client': pd.Series([], dtype='float'),
                       'class_per_client': pd.Series([], dtype='float')})
    y_size = []
    all_classes = set()
    for i, user in enumerate(users):
        ds = data[user]
        assert isinstance(ds, Dataset)
        num_data = len(ds)
        y_unique = set([y for _, y in ds])
        y_size.append(num_data)
        # y 个类别的数量
        # y_size.append(len(y_unique))
        all_classes.update(y_unique)
        #
        df = df.append({'samples_per_client': num_data, 'class_per_client': len(y_unique)}, ignore_index=True)
    print(title)
    print('Number of clients: {}, samples: {}, classes: {}\n'.format(len(users), sum(y_size), len(all_classes)))
    print(df.describe())
