import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, download_and_extract_archive
import re
import os

PREFIX = os.path.dirname(os.path.realpath(__file__))
EMBEEDING_PREFIX = os.path.join(PREFIX, 'data', 'embeddings')

for i in [os.path.join(PREFIX, 'data'), EMBEEDING_PREFIX]:
    if not os.path.exists(i):
        os.mkdir(i)


ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def letter_to_index(letter):
    """
    字母转换为索引
    :param letter:
    :return:
    """
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(word):
    """
    将某个单词转换为 index
    :param word:
    :return:
    """
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


# ------------------------
# 可用于Sent40, Shakespeare


def split_line(line):
    '''split given line/phrase into list of words

    Args:
        line: string representing phrase to be split

    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    '''returns index of given word based on given lookup dictionary

    returns the length of the lookup dictionary if word not found

    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    '''
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices

    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer
    representing unknown index to returned list until the list's length is
    max_words

    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list

    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line)  # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id] * (max_words - len(indl))
    return indl



WordEmbedding = namedtuple('WordEmbedding', 'word2index index2word unk_index num_vocabulary embedding')


class Sent140(Dataset):

    WORD_EMBEDDING = None
    WORD_EMBEDDING_URLS = {
        'glove.6B': 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.6B.zip'
    }

    def __init__(self, data, options):
        """
        这个类在读取的 pkl 为实际的数据的时候用于将 dict 格式的数据转换为 Tensor
        :param data:
        :param labels:
        """
        super(Sent140, self).__init__()
        # 句子的序列, 长度为 80
        # 标记, 一个字符, 需要给出他的 index
        # 先试试能否一开始预处理, 这个和 tf 版本不一样, label 不需要 one-hot
        if Sent140.WORD_EMBEDDING is None:
            Sent140.WORD_EMBEDDING = Sent140.load_word_embedding(embedding_name='glove.6B', embedding_dim=300)
        x_batch = [e[4] for e in data['x']]
        x_batch = [line_to_indices(e, self.WORD_EMBEDDING.word2index, max_words=25) for e in x_batch]
        x_batch = np.array(x_batch, dtype=np.int64)
        self.sentences = x_batch
        # 可以先处理句子
        self.target = np.asarray([1 if e == '4' else 0 for e in data['y']], dtype=np.int64)

    @staticmethod
    def load_word_embedding(embedding_name, embedding_dim: int=300) -> WordEmbedding:
        assert Sent140.WORD_EMBEDDING is None, '只能加载一次'
        extract_root = os.path.join(EMBEEDING_PREFIX, embedding_name)
        if embedding_name.startswith('glove'):
            target_filename = os.path.join(extract_root, embedding_name + '.{}d.txt'.format(embedding_dim))
        else:
            raise ValueError('不能加载 {} 对应的信息'.format(embedding_name))

        if not os.path.exists(target_filename):
            tmp_root = os.path.join(EMBEEDING_PREFIX, 'tmp')
            download_and_extract_archive(url=Sent140.WORD_EMBEDDING_URLS[embedding_name], download_root=tmp_root,
                                         extract_root=extract_root, remove_finished=True)

        # 加载词向量, 这里默认用 GLOVE
        print('使用词向量文件: {}'.format(target_filename))
        with open(target_filename, 'r') as inf:
            lines = inf.readlines()
        # lines[0] 词, 后面是向量的每个维度
        lines = [l.split() for l in lines]
        # 所有的词
        index2word = [l[0] for l in lines]
        # 词对应的向量列表
        emb_floats = [np.asarray([float(n) for n in l[1:]], dtype=np.float32) for l in lines]
        # 默认最后一个维度作为 UNK, 这里的词嵌入增加了最后一维度, 作为 UNK, 但是词汇表数量是少一个的
        emb_floats.append(np.zeros([embedding_dim], dtype=np.float32))  # for unknown word
        #
        embedding = np.stack(emb_floats, axis=0)
        #
        word2index = {v: k for k, v in enumerate(index2word)}
        return WordEmbedding(word2index=word2index,
                             index2word=index2word,
                             embedding=embedding,
                             unk_index=len(index2word),
                             num_vocabulary=len(index2word))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.sentences[index], self.target[index]


