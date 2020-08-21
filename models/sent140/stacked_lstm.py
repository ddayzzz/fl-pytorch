import torch
from torch import nn
import torch.nn.functional as F
from dataset.sent140.sent140 import Sent140


class Model(nn.Module):

    def __init__(self, options, seq_len, num_classes, n_hidden, embedding_dim):
        super(Model, self).__init__()
        # 提前初始化
        if Sent140.WORD_EMBEDDING is None:
            Sent140.WORD_EMBEDDING = Sent140.load_word_embedding(embedding_name='glove.6B', embedding_dim=embedding_dim)
        # 定义参数
        self.input_shape = [seq_len]
        self.input_type = 'index'  # 输入是 index
        # 注意 UNK
        # [B(SEQ_LEN), 300]
        # 这是不可以学习的参数, 使用 buffer, 避免作为 parameter 被监控到!!
        # self.embedding_layer = nn.Embedding(num_embeddings=Sent140.WORD_EMBEDDING.num_vocabulary + 1,
        #                                     embedding_dim=embedding_dim,
        #                                     sparse=False,
        #                                     _weight=torch.from_numpy(Sent140.WORD_EMBEDDING.embedding))
        embedding = torch.from_numpy(Sent140.WORD_EMBEDDING.embedding)
        self.register_buffer('embedding_weight', embedding)
        # 不计算梯度
        # self.embedding_layer.weight.requires_grad = False
        # 输入: (seq_len, batch, input_size), hx(2,batch,hidden_size)
        # 输出: (seq_len, batch, num_directions * hidden_size), 如果 batch_first == True, 交换 0, 1
        self.stacked_lstm = nn.LSTM(input_size=embedding_dim,
                                    hidden_size=n_hidden,
                                    num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(in_features=n_hidden, out_features=128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x = self.embedding_layer(x)
        x = F.embedding(x, weight=self.embedding_weight)
        # 将 embedding 前任嵌入的数据转换, 这里不传入 hidden, LSTM 自动处理
        x, _ = self.stacked_lstm(x)
        # 预测是那个人物, 用最后一句话?
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.fc2(x)
        return x