# @ Time : 2022/4/9,16:36
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description: 将算法运行过程中的各个阶段进行可视化输出的库

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np


def print3D(df_data):
    """
    可视化  输出3D点云数据
    :param df_data:点云数据
    :return: None
    """
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(12, 10), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    X = df_data['x'].values
    Y = df_data['y'].values
    Z = df_data['z'].values
    V = df_data['v'].values
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 10)
    ax.set_zlim(-6, 6)

    my_cmap = plt.get_cmap('ocean')
    plt.tick_params(labelsize=25)
    sctt = ax.scatter(X, Y, Z, s=20, marker='o', c=V, cmap=my_cmap)

    ax.set_zlabel('Z', fontdict={'size': 25, 'color': 'black'}, labelpad=5)
    ax.set_ylabel('Y', fontdict={'size': 25, 'color': 'black'}, labelpad=20)
    ax.set_xlabel('X', fontdict={'size': 25, 'color': 'black'}, labelpad=20)
    fc = fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=10)
    fc.set_label('径向速度', fontsize=20, x=20, y=0.5)

    plt.title('3D点云示意图', size=30)
    plt.show()
    pass


def print2D(df_data):
    """
    可视化 输出3D点云数据的三视图
    :param df_data:点云数据
    :return: None
    """

    fig = plt.figure(figsize=(12, 10), dpi=80)
    X = df_data['x']
    Y = df_data['y']
    Z = df_data['z']

    ax = plt.subplot(2, 2, 1)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 10)
    plt.tick_params(labelsize=30)
    plt.scatter(X, Y, s=10, edgecolor='black', marker='o')

    ax = plt.subplot(2, 2, 2)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    plt.tick_params(labelsize=30)
    plt.yticks([-6, -4, -2, 0, 2, 4, 6])
    plt.scatter(X, Z, s=10, edgecolor='black', marker='o')

    ax = plt.subplot(2, 2, 3)
    ax.set_xlim(0, 10)
    ax.set_ylim(-6, 6)
    plt.tick_params(labelsize=30)
    plt.yticks([-6, -4, -2, 0, 2, 4, 6])
    plt.xticks([int(0), 2.5, 5, 7.5, 10])
    plt.scatter(Y, Z, s=10, edgecolor='black', marker='o')

    plt.show()
    pass


def print_velocity(df_data):
    """
    可视化 绘制速度分布图
    :param df_data:点云数据
    :return: None
    """

    fig = plt.figure(figsize=(12, 10), dpi=80)
    X = df_data['z']
    Y = df_data['v']

    plt.tick_params(labelsize=40)
    plt.scatter(Y, X, s=20, edgecolor='black', marker='o')
    plt.show()
    pass


def print_distance(df_data):
    """
    可视化 绘制距离分布图
    :param df_data:点云数据
    :return: None
    """

    fig = plt.figure(figsize=(12, 10), dpi=80)
    X = df_data['frame']
    Y = df_data['distance']

    plt.tick_params(labelsize=30)
    plt.scatter(X, Y, s=20, edgecolor='black', marker='o')
    plt.show()
    pass


def print_angle(df_data):
    """
    可视化 绘制角度分布图
    :param df_data:点云数据
    :return: None
    """

    fig = plt.figure(figsize=(12, 10), dpi=80)
    X = df_data['x']
    Y = df_data['azimuth']
    Z = df_data['elevation']

    ax = plt.subplot(1, 2, 1)
    plt.tick_params(labelsize=20)
    ax.set_ylim(-90, 90)
    plt.scatter(X, Y, s=20, edgecolor='black')

    ax = plt.subplot(1, 2, 2)
    plt.tick_params(labelsize=20)
    ax.set_ylim(-90, 90)
    plt.scatter(X, Z, s=20, edgecolor='black')

    plt.show()
    pass


def print_points(path):
    """
    可视化 统计行人数量对数据长度的影响
    :param path: 数据路径
    :return: None
    """

    p1 = []
    p2 = []
    p3 = []
    p4 = []

    # 计算长度
    label_list = [1, 2, 3, 4]
    for index in range(100):
        for label in label_list:
            df_data = pd.read_csv(f'{path}/{index}_{label}.csv')
            length = len(df_data)
            if label == 1:
                p1.append(length)
            elif label == 2:
                p2.append(length)
            elif label == 3:
                p3.append(length)
            else:
                p4.append(length)

    mean_1 = sum(p1) / len(p1)
    mean_2 = sum(p2) / len(p2)
    mean_3 = sum(p3) / len(p3)
    mean_4 = sum(p4) / len(p4)

    X = [1, 2, 3, 4]
    Y = [mean_1, mean_2, mean_3, mean_4]
    Z = [mean_1, mean_2 / 2, mean_3 / 3, mean_4 / 4]

    # 绘制图像
    fig = plt.figure(figsize=(12, 10), dpi=80)
    mpl.rcParams['font.family'] = 'SimHei'

    ax = plt.subplot(1, 2, 1)
    plt.xticks([1, 2, 3, 4])
    plt.yticks([400, 500, 600, 700, 800, 900])
    plt.grid(linestyle='-.')
    plt.tick_params(labelsize=20)
    plt.plot(X, Y, marker='^', markersize=15)

    ax = plt.subplot(1, 2, 2)
    plt.xticks([1, 2, 3, 4])
    plt.yticks([200, 250, 300, 350, 400, 450])
    plt.grid(linestyle='-.')
    plt.tick_params(labelsize=20)
    plt.plot(X, Z, marker='^', markersize=15, c='g')

    plt.show()
    pass


def print_sff(df_suspicious_points):
    """
    可视化 绘制可疑帧域
    :param df_suspicious_points: 可以帧域列表
    :return: None
    """

    # 下列选项为将所有的SFF绘制在一张图上、或分开绘制
    # fig = plt.figure(figsize=(12, 10), dpi=80)
    # ax = fig.add_subplot(111, projection='3d')

    # 生成颜色列表
    colors = ['black', 'blue', 'red', 'orange', 'green', 'gray']
    index = 0

    for df_suspicious_point in df_suspicious_points:
        fig = plt.figure(figsize=(12, 10), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        X = df_suspicious_point['x'].values
        Y = df_suspicious_point['y'].values
        Z = df_suspicious_point['z'].values
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 10)
        ax.set_zlim(-6, 6)
        ax.scatter(X, Y, Z, s=15, marker='o', label=index, c=colors[index])
        plt.yticks([0, 5, 10])
        plt.tick_params(labelsize=50)
        plt.show()
        index += 1

    # plt.show()
    pass


def print_snake(snake_list):
    """
    可视化 绘制轨迹分割结果
    :param snake_list: 轨迹列表
    :return: None
    """

    # 下列选项为将所有的SFF绘制在一张图上、或分开绘制
    # fig = plt.figure(figsize=(12, 10), dpi=80)
    # ax = fig.add_subplot(111, projection='3d')

    # 颜色列表
    colors = ['black', 'red']
    index = 0

    for snake in snake_list:
        fig = plt.figure(figsize=(12, 10), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        X = snake['x'].values
        Y = snake['y'].values
        Z = snake['z'].values
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 10)
        ax.set_zlim(-6, 6)
        ax.scatter(X, Y, Z, s=15, marker='o', label=index, c=colors[index])
        plt.tick_params(labelsize=20)
        plt.show()
        index += 1

    # plt.show()
    pass


def print_segment(segment_list):
    """
    可视化 绘制轨迹分割的结果 共5个部分
    :param segment_list:
    :return: None
    """

    fig = plt.figure(figsize=(12, 10), dpi=80)

    ax = plt.subplot(2, 3, 1)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    plt.tick_params(labelsize=20)
    plt.scatter(segment_list[0]['x'], segment_list[0]['z'], s=10, edgecolor='black', marker='o')

    ax = plt.subplot(2, 3, 2)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    plt.tick_params(labelsize=20)
    plt.scatter(segment_list[1]['x'], segment_list[1]['z'], s=10, edgecolor='black', marker='o')

    ax = plt.subplot(2, 3, 3)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    plt.tick_params(labelsize=20)
    plt.scatter(segment_list[2]['x'], segment_list[2]['z'], s=10, edgecolor='black', marker='o')

    ax = plt.subplot(2, 3, 4)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    plt.tick_params(labelsize=20)
    plt.scatter(segment_list[3]['x'], segment_list[3]['z'], s=10, edgecolor='black', marker='o')

    ax = plt.subplot(2, 3, 5)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    plt.tick_params(labelsize=20)
    plt.scatter(segment_list[4]['x'], segment_list[4]['z'], s=10, edgecolor='black', marker='o')

    plt.show()
    pass


def print_density(density):
    """
    可视化 绘制密度图
    :param density:密度图Numpy
    :return: None
    """

    ax = plt.gca()
    plt.imshow(density, cmap=plt.cm.gray)
    ax.invert_yaxis()
    plt.colorbar()
    plt.tick_params(labelsize=20)
    plt.show()
    pass


def print_in_time(df_data):
    """
    可视化 逐帧绘制点云数据
    :param df_data:点云数据
    :return: None
    """
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(12, 10), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    plt.title('3D点云动态示意图（去噪前）', size=20)
    my_cmap = plt.get_cmap('ocean')
    # plt.axis('off')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    plt.grid(True)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 10)
    ax.set_zlim(-6, 6)

    ax.set_zlabel('Z', fontdict={'size': 25, 'color': 'black'}, labelpad=5)
    ax.set_ylabel('Y', fontdict={'size': 25, 'color': 'black'}, labelpad=20)
    ax.set_xlabel('X', fontdict={'size': 25, 'color': 'black'}, labelpad=20)

    frames = df_data['frame'].drop_duplicates()  # 去除重复值

    for frame in frames:
        X = df_data[df_data['frame'] == frame]['x'].values
        Y = df_data[df_data['frame'] == frame]['y'].values
        Z = df_data[df_data['frame'] == frame]['z'].values
        V = df_data[df_data['frame'] == frame]['v'].values

        sctt = ax.scatter(X, Y, Z, s=20, marker='o', c=V, cmap=my_cmap)

        # fc = fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=10)
        # fc.set_label('径向速度', fontsize=20, x=20, y=0.5)

        plt.pause(0.3)

    plt.show()
    pass


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots()
    plt.tick_params(labelsize=16)
    y1 = [1, 1, 0.98, 1, 1]
    y2 = [0.96, 1, 0.96, 0.98, 0.98]
    y3 = [0.94, 0.94, 0.96, 0.92, 0.98]
    y4 = [0.92, 0.96, 0.94, 0.92, 0.94]
    y5 = [0.94, 0.94, 0.92, 0.96, 0.94]

    labels = ['第一次实验', '第二次实验', '第三次实验', '第四次实验', '第五次实验']
    x = np.arange(5)
    width = 0.1
    rects1 = ax.bar(x - width * 2, y1, width, label='0人')
    rects2 = ax.bar(x - width + 0.01, y2, width, label='1人')
    rects3 = ax.bar(x + 0.02, y3, width, label='2人')
    rects4 = ax.bar(x + width + 0.03, y4, width, label='3人')
    rects5 = ax.bar(x + width * 2 + 0.04, y5, width, label='4人')

    ax.set_ylabel('准确率', fontsize=20)
    ax.set_xlabel('行人个数', fontsize=20)
    ax.set_title('算法测试结果', fontsize=20)
    ax.set_ylim(0.50, 1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=20)

    fig.tight_layout()

    plt.show()
