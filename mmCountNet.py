# @ Time : 2022/4/9,19:27
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description: mmCountNet 卷积神经网络分类器

import pandas as pd
import torch
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def same_seed(seed):
    """
    固定时间种子
    :param seed: 时间种子
    :return:  None
    """

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        pass
    pass


class mmDataset(Dataset):
    # 定义数据集
    def __init__(self, path, files=None):
        super(mmDataset).__init__()
        self.path = path
        self.files = os.listdir(path)
        if files:
            self.files = files
        pass

    def __getitem__(self, item):
        fname = self.files[item]
        segment = np.array(pd.read_csv(os.path.join('mmCountData/data_train/seg_csv/', fname), index_col=0))
        segment = segment.reshape(5, 64, 64)
        segment = torch.FloatTensor(segment)
        # 根据文件名获取标签label
        label = int(fname.split('.')[0].split('_')[-1])
        return segment, label

    def __len__(self):
        return len(self.files)


class mmCountNet(nn.Module):
    # 定义模型
    def __init__(self, input_dim=5, output_dim=5):
        super(mmCountNet, self).__init__()

        self.cnn = nn.Sequential(
            # 输入为 32 * 5 * 64 * 64  batch_size = 32
            # 第一个layer
            nn.Conv2d(input_dim, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [32, 16, 32, 32]

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [32, 32, 16, 16]

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [32, 64, 8, 8]

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [32, 128, 4, 4]
        )

        self.fc = nn.Sequential(
            # 输入为 512 * 4 * 4
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, segment):
        out = self.cnn(segment)
        # 将卷积的结果拉伸成一个vector 作为全连接层的输入
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


def trainer(train_loader, valid_loader, model, config, device):
    """
    定义训练过程
    :param train_loader: 训练集
    :param valid_loader: 验证集
    :param model: 模型
    :param config: 参数列表
    :param device: GPU or CPU
    :return: 模型训练过程记录
    """

    # 定义损失函数、迭代器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)

    # 创建model文件夹
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    # 定义迭代次数， 最佳准确率
    n_epochs, best_acc = config['n_epochs'], 0

    # 定义训练过程记录
    loss_record_train = []
    acc_record_train = []
    loss_record_valid = []
    acc_record_valid = []

    # 定义迭代
    for epoch in range(n_epochs):

        # training
        model.train()  # 转换为train模式

        train_loss_record = []  # 记录损失函数的值
        train_acc_record = []  # 记录准确率

        for i, batch in enumerate(tqdm(train_loader)):
            segments, labels = batch
            segments = segments.to(device)
            labels = labels.to(device)

            # 梯度下降五步走
            optimizer.zero_grad()
            y_preds = model(segments)
            loss = criterion(y_preds, labels)
            loss.backward()
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)  # 剪裁梯度范数以进行稳定训练。
            optimizer.step()

            # 记录数据
            acc = (y_preds.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss_record.append(loss.item())
            train_acc_record.append(acc)
            pass

        train_loss = sum(train_loss_record) / len(train_loss_record)
        train_acc = sum(train_acc_record) / len(train_acc_record)

        # validating
        model.eval()
        valid_loss_record = []  # 记录损失函数的值
        valid_acc_record = []  # 记录准确率

        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                segments, labels = batch
                segments = segments.to(device)
                labels = labels.to(device)

                # 预测
                y_preds = model(segments)
                loss = criterion(y_preds, labels)

                # 记录数据
                acc = (y_preds.argmax(dim=-1) == labels.to(device)).float().mean()
                valid_loss_record.append(loss.item())
                valid_acc_record.append(acc)
                pass

            valid_loss = sum(valid_loss_record) / len(valid_loss_record)
            valid_acc = sum(valid_acc_record) / len(valid_acc_record)

        # 输出本次结果
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Valid Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, n_epochs, train_acc, train_loss, valid_acc, valid_loss
        ))

        loss_record_train.append(train_loss)
        acc_record_train.append(train_acc)
        loss_record_valid.append(valid_loss)
        acc_record_valid.append(valid_acc)

        # 记录最好模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(config['model_path'], 'model.ckpt'))
            print(f"Best model found at epoch {epoch}, best acc{best_acc},saving model")
            pass

    return loss_record_train, acc_record_train, loss_record_valid, acc_record_valid


def predictor(density, model, device):
    """
    定义预测函数 输入单个数据，返回一个预测结果
    :param density: 密度图
    :param model: 模型
    :param device: GPU or CPU
    :return: 预测结果
    """

    segment = density.reshape(1, 5, 64, 64)
    segment = torch.FloatTensor(segment)

    with torch.no_grad():
        segment = segment.to(device)
        y_pred = model(segment)

    best_y = y_pred.argmax(1)
    return int(best_y)


def show_trainer(loss_record_train, acc_record_train, loss_record_valid, acc_record_valid):
    """
    训练过程可视化
    :param loss_record_train: 训练集loss
    :param acc_record_train: 训练集acc
    :param loss_record_valid: 验证集loss
    :param acc_record_valid: 验证集acc
    :return: None
    """

    # 绘制图像
    fig = plt.figure(figsize=(12, 10), dpi=80)
    mpl.rcParams['font.family'] = 'SimHei'

    # 将数据从GPU转移至CPU
    loss_record_train = torch.tensor(loss_record_train, device='cpu')
    acc_record_train = torch.tensor(acc_record_train, device='cpu')
    loss_record_valid = torch.tensor(loss_record_valid, device='cpu')
    acc_record_valid = torch.tensor(acc_record_valid, device='cpu')

    X = [i for i in range(30)]
    plt.grid(linestyle='-.')
    plt.tick_params(labelsize=50)
    # plt.plot(X, loss_record_train, marker='s', markersize=30, markerfacecolor='none', label='训练集')
    plt.plot(X, acc_record_train, marker='^', markersize=30, markerfacecolor='none', label='训练集')
    # plt.plot(X, loss_record_valid, marker='s', markersize=30, markerfacecolor='r',label='验证集')
    plt.plot(X, acc_record_valid, marker='^', markersize=30, markerfacecolor='b', label='验证集')

    plt.legend(fontsize=40)
    plt.show()
    pass


def show_mmCountNet():
    """
    模型结构可视化 生成一个tex文件
    :return: None
    """

    sys.path.append('./PYCORE_FILE')

    # 定义网络
    arch = [
        to_head('.'),
        to_cor(),
        to_begin(),

        # 输入密度图
        to_input('density.jpg', height=5, width=5),

        # 添加卷积层conv1
        to_ConvConvRelu(name='conv1', s_filer=64, n_filer=(16, 16), offset="(0,0,0)", to="(0,0,0)", width=(2, 2),
                        height=40, depth=40, caption='Conv1'),
        to_Pool(name="pool_b1", offset="(0,0,0)", to="(conv1-east)", width=1, height=32, depth=32, opacity=0.5),

        # 添加卷积层conv2
        to_ConvConvRelu(name='conv2', s_filer=32, n_filer=(32, 32), offset="(2,0,0)", to="(pool_b1-east)",
                        width=(3.5, 3.5), height=32, depth=32, caption='Conv2'),
        to_Pool(name="pool_b2", offset="(0,0,0)", to="(conv2-east)", width=1, height=25, depth=25, opacity=0.5),
        to_connection('pool_b1', 'conv2'),

        # 添加卷积层conv3
        to_ConvConvRelu(name='conv3', s_filer=16, n_filer=(64, 64), offset="(2,0,0)", to="(pool_b2-east)",
                        width=(4.5, 4.5), height=25, depth=25, caption='Conv3'),
        to_Pool(name="pool_b3", offset="(0,0,0)", to="(conv3-east)", width=1, height=16, depth=16, opacity=0.5),
        to_connection('pool_b2', 'conv3'),

        # 添加卷积层conv4
        to_ConvConvRelu(name='conv4', s_filer=8, n_filer=(128, 128), offset="(2,0,0)", to="(pool_b3-east)",
                        width=(5.5, 5.5), height=16, depth=16, caption='Conv4'),
        to_Pool(name="pool_b4", offset="(0,0,0)", to="(conv4-east)", width=1, height=10, depth=10, opacity=0.5),
        to_connection('pool_b3', 'conv4'),

        # 衔接卷积层与全连接层
        to_SoftMax("FC_1", 2048, "(2,0,0)", to="(pool_b4-east)", width=2, height=40, depth=2),
        to_connection('pool_b4', 'FC_1'),

        # 添加全连接层fc1
        to_SoftMax("FC_2", 1024, "(1,0,0)", to="(FC_1-east)", width=2, height=32, depth=2),
        to_connection('FC_1', 'FC_2'),

        # 添加全连接层fc2
        to_SoftMax("FC_3", 512, "(1,0,0)", to="(FC_2-east)", width=2, height=25, depth=2),
        to_connection('FC_2', 'FC_3'),

        # 添加全连接层fc3
        to_SoftMax("softmax", 5, "(1,0,0)", to="(FC_3-east)", width=2, height=5, depth=2),
        to_connection('FC_3', 'softmax'),

        # 结束
        to_end()
    ]

    # 生成tex文件 用LaTeX运行即可得到结果
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')
    pass


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        # 数据参数
        'path': 'mmCountData/data_train/seg_csv',
        'train_ratio': 0.8,

        # 模型参数
        'seed': 1127,
        'input_dim': 64,
        'output_dim': 3,
        'batch_size': 32,
        'n_epochs': 30,
        'learning_rate': 1e-5,

        # 存储参数
        'model_path': 'models/'
    }

    # 固定时间种子
    same_seed(config['seed'])

    # 分割训练集和验证集
    dataset = mmDataset(config['path'])
    train_len = int(config['train_ratio'] * len(dataset))
    lengths = [train_len, len(dataset) - train_len]
    train_set, valid_set = random_split(dataset, lengths)

    # 获得dataloader
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    # 运行model
    model = mmCountNet().to(device)
    loss_record_train, acc_record_train, loss_record_valid, acc_record_valid = trainer(train_loader, valid_loader,
                                                                                       model, config, device)

    # 可视化模型训练过程
    # show_trainer(loss_record_train, acc_record_train, loss_record_valid, acc_record_valid)

    print(loss_record_train, loss_record_valid)
    print(acc_record_train, acc_record_valid)

    # 展示模型
    # show_mmCountNet()
