# @ Time : 2022/4/9,16:06
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description: 将mmWave Sensor采集到的数据进行各种处理的库

import math
import shutil
import time
import torch
from random import shuffle

from visualizer import *
from mmCountNet import mmCountNet, predictor
from TI_FILE.mmw_demo_example_script import *

from sklearn import svm, cluster, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def convert_name(prefix, label_list):
    """
    批量重命名毫米波雷达采集到的dat文件
    :param prefix: 文件路径
    :param label_list: 标签列表
    :return: None
    """

    for label in label_list:
        # 获取当前标签下所有文件名称列表
        data_list = os.listdir(f'{prefix}/{label}')
        index = 0

        # 重命名
        for data in data_list:
            os.rename(f'{prefix}/{label}/{data}', f'{prefix}/{label}/{index}_{label}.dat')
            index += 1
    pass


def convert_data(prefix, label_list, save_path):
    """
    批量将dat文件转化为csv文件，并为其打上标签
    :param prefix: 数据路径
    :param label_list: 标签列表
    :param save_path: 保存路径
    :return: None
    """
    # 根据label遍历所有数据

    for label in label_list:
        # 获取当前标签下所有文件名称列表
        data_list = os.listdir(f'{prefix}/{label}')
        index = 0
        # 解析数据
        for data in data_list:
            Analytical_data(f'./{prefix}/{label}/{data}', index, label, save_path)
            index += 1
        pass
    pass


def find_sff(df_data):
    """
    寻找可疑帧域(Suspicious Frame Field)
    :param df_data: 源数据
    :return: SSF列表
    """

    min_points = 50
    min_frames = 30

    # 遍历寻找可疑帧域
    startFrame = 0  # 起始帧
    endFrame = 0  # 结束帧
    count = 0  # ssf长度
    ssf_list = []
    df_suspicious_points = pd.DataFrame(
        columns=['index', 'frame', 'x', 'y', 'z', 'v', 'distance', 'azimuth', 'elevation'])

    for data in df_data.itertuples(index=False):
        if not startFrame:
            startFrame = endFrame = data[0]  # 记录开始帧

        if data[0] - endFrame > min_frames:  # 即持续1秒以上未读取到数据 则认为本次移动已结束
            if df_suspicious_points.shape[0] > min_points:
                print(
                    "➡|   [{},{},{}]  |".format(str(startFrame).zfill(4), str(endFrame).zfill(4), str(count).zfill(4)))
                ssf_list.append(df_suspicious_points)

            startFrame = endFrame = data[0]
            df_suspicious_points = df_suspicious_points.drop(index=df_suspicious_points.index)  # 清空df_suspicious_points
            count = 0

        df_temp = pd.DataFrame([[count, data[0], data[2], data[3], data[4], data[5], data[6], data[7], data[8]]],
                               columns=['index', 'frame', 'x', 'y', 'z', 'v', 'distance', 'azimuth', 'elevation'])
        df_suspicious_points = pd.concat([df_suspicious_points, df_temp])
        count += 1
        endFrame = data[0]

    if df_suspicious_points.shape[0] > min_points:
        ssf_list.append(df_suspicious_points)
        print("➡|   [{},{},{}]  |".format(str(startFrame).zfill(4), str(endFrame).zfill(4), str(count).zfill(4)))

    return ssf_list


def physical_denoise(df_data):
    """
    数据去噪-物理去噪 根据物理特征去除数据中的噪声
    :param df_data: 点云数据
    :return: 物理去噪后的点云数据
    """

    # 筛选x、y、z 去除反射噪声
    df_data = df_data[(-3.2 < df_data['z']) & (df_data['z'] < 3.2)]
    df_data = df_data[(-3.2 < df_data['x']) & (df_data['x'] < 3.2)]
    df_data = df_data[(0 < df_data['y']) & (df_data['y'] < 10)]

    # 筛选v 去除非行人噪声
    df_data = df_data[(df_data['v'] > 0.5) | (df_data['v'] < -0.5)]
    df_data = df_data[(-20 < df_data['elevation']) & (df_data['elevation'] < 60)]

    return df_data


def cluster_denoise(df_data):
    """
    数据去噪-聚类去噪 保留核心数据 去除孤立数据
    :param df_data: 点云数据
    :return: 聚类去噪后的点云数据
    """

    # 自定义距离度量,减小 y 轴的影响
    def distance(point_A, point_B, alpha=0.2):
        return math.sqrt((point_A[0] - point_B[0]) * (point_A[0] - point_B[0]) + alpha * (point_A[1] - point_B[1]) * (
                point_A[1] - point_B[1]) + (point_A[2] - point_B[2]) * (point_A[2] - point_B[2]))

    # 准备点云数据, X的长度为检测到的点的个数
    X = np.array(df_data.loc[:, ['x', 'y', 'z']])

    # 准备模型训练相关数据
    epsilons, scores, clusters = np.linspace(0.8, 1.5, 10), [], []  # np.linspace是序列生成器，以均匀的步长生成数字序列，且不会损失浮点数京都

    # 遍历所有半径，找到得分最高的参数
    for epsilon in epsilons:
        cls = cluster.DBSCAN(eps=epsilon, min_samples=20, metric=lambda a, b: distance(a, b))
        cls.fit(X)
        if len(np.unique(cls.labels_)) == 1:
            scores.append(0)
            clusters.append(cls)
            continue
        score = metrics.silhouette_score(X, cls.labels_, sample_size=len(X), metric='euclidean')
        scores.append(score)
        clusters.append(cls)

    # 分析训练结果
    scores = np.array(scores)
    best_index = scores.argmax()

    # 利用最优模型进行聚类
    best_cls = clusters[best_index]

    # 获得孤立样本、外周样本、核心样本
    core_mask = np.zeros(len(X), dtype=bool)
    core_mask[best_cls.core_sample_indices_] = True  # 将核心样本的索引设置为True

    # 保留核心样本
    return df_data[core_mask]


def frame_segment(df_data):
    """
    分割帧,转化为segment
    :param df_data: 点云数据
    :return:切片列表
    """

    # 将每组数据切割为5份 可以保留时间的特征 还可以解决数据不足的问题
    num = int(len(df_data) / 7)
    # 交叉获取 弥补点数不足的情况
    segment_list = []
    for start in range(0, len(df_data), num):
        if start + num * 3 > len(df_data):
            break

        segment = df_data.iloc[start:start + num * 3].loc[:, ['x', 'z']]
        segment_list.append(segment)
        pass

    return segment_list


def convert_Numpy(segment_list, index=0, label=0, save_path=None):
    """
    转化为64 * 64 的Numpy
    图片按照 6.4m * 6.4m 每0.1m * 0.1m一个格子
    :param segment_list:切片列表
    :param index: 数据序号
    :param label: 数据标签
    :param save_path: 保存路径
    :return: 密度图
    """

    # 密度图
    density_list = []
    density_flip_list = []
    for segment in segment_list:
        density = np.zeros((64, 64), dtype=int)

        for point in segment.itertuples(index=False):
            y = int(point[0] * 10) + 32
            x = int(point[1] * 10) + 32
            density[x, y] += 1

        density_list.append(density)
        density = np.flip(density, axis=1)
        density_flip_list.append(density)

    density_numpy_A = np.concatenate(
        (density_list[0], density_list[1], density_list[2], density_list[3], density_list[4]), axis=0)
    density_numpy_B = np.concatenate(
        (density_list[4], density_list[3], density_list[2], density_list[1], density_list[0]), axis=0)
    density_numpy_C = np.concatenate(
        (density_flip_list[0], density_flip_list[1], density_flip_list[2], density_flip_list[3], density_flip_list[4]),
        axis=0)
    density_numpy_D = np.concatenate(
        (density_flip_list[4], density_flip_list[3], density_flip_list[2], density_flip_list[1], density_flip_list[0]),
        axis=0)
    if save_path:
        pd.DataFrame(density_numpy_A).to_csv(f'{save_path}/density_{index}_{label}.csv')
        pd.DataFrame(density_numpy_B).to_csv(f'{save_path}/density_{index + 1}_{label}.csv')
        pd.DataFrame(density_numpy_C).to_csv(f'{save_path}/density_{index + 2}_{label}.csv')
        pd.DataFrame(density_numpy_D).to_csv(f'{save_path}/density_{index + 3}_{label}.csv')
    return density_numpy_A


def snake_cluster(df_data):
    """
    划分数据非同时移动者的移动轨迹
    :param df_data: 点云数据
    :return: 轨迹列表
    """

    min_points = 50
    max_distance = 0.8

    snake_list = []
    snake_head_list = []
    snake_ahead_list = []

    # 计算两点之间的距离
    def get_distance(point_A, point_B):
        return math.sqrt((point_A[1] - point_B[1]) * (point_A[1] - point_B[1]) + (point_A[2] - point_B[2]) * (
                point_A[2] - point_B[2]))

    frames = df_data['frame'].drop_duplicates()  # 去除重复值

    # 初始化snake, 保持蛇头长度为10帧
    frame_seg = frames[0:10]
    points = df_data.loc[df_data['frame'].isin(list(frame_seg))][['frame', 'x', 'y', 'z']]
    snake_body = pd.DataFrame(columns=['frame', 'x', 'y', 'z'])
    snake_body = pd.concat([snake_body, points])
    snake_head = snake_body

    snake_list.append(snake_body)
    snake_head_list.append(snake_head)
    snake_ahead_list.append(pd.DataFrame(columns=['frame', 'x', 'y', 'z']))
    last_frame = frames.iloc[9]

    # 遍历当前帧的点 和 snake里的点
    for frame in frames.iloc[10:]:
        # 遍历每一帧
        points = df_data[df_data['frame'] == frame][['frame', 'x', 'y', 'z']]

        for point_A in points.itertuples(index=False):
            index = 0
            min_distance = 100
            min_index = 0

            for snake_head in snake_head_list:
                # print(len(snake_head))
                for point_B in snake_head.itertuples(index=False):
                    if get_distance(point_A, point_B) < min_distance:
                        min_distance = get_distance(point_A, point_B)
                        min_index = index
                index += 1

            # 判断该点的轨迹归属
            if min_distance < abs(last_frame - frame) * max_distance:
                snake_ahead_list[min_index].loc[len(snake_ahead_list[min_index])] = point_A
            else:

                # 发现新的贪吃蛇
                snake_body_temp = pd.DataFrame(columns=['frame', 'x', 'y', 'z'])
                snake_body_temp.loc[len(snake_body_temp)] = point_A
                snake_head_temp = snake_body_temp

                snake_list.append(snake_body_temp)
                snake_head_list.append(snake_head_temp)
                snake_ahead_list.append(pd.DataFrame(columns=['frame', 'x', 'y', 'z']))

        #  贪吃蛇更新
        last_frame = frame
        for index in range(len(snake_list)):

            snake_list[index] = pd.concat([snake_list[index], snake_ahead_list[index]])
            snake_head_list[index] = pd.concat([snake_head_list[index], snake_ahead_list[index]])
            snake_ahead_list[index] = snake_ahead_list[index].drop(index=snake_ahead_list[index].index)

            # 蛇头不一次一更， 5帧一更新更好
            if len(snake_head_list[index]['frame'].drop_duplicates()) > 10:
                frame_seg = snake_head_list[index]['frame'].drop_duplicates()[-10:]
                snake_head_list[index] = snake_head_list[index].loc[
                    snake_head_list[index]['frame'].isin(list(frame_seg))]

    # 去除过短的蛇
    ans_list = []
    for snake in snake_list:
        if len(snake) > min_points:
            ans_list.append(snake)

    return ans_list


def people_count(prefix):
    """
    功能函数 实现批量从dat输入 到统计结果输出
    :param prefix: 待处理文件所在路径
    :return: None
    """

    # 建立临时文件文件夹temp
    if os.path.isdir(f'{prefix}/temp'):
        shutil.rmtree(f'{prefix}/temp')
    os.makedirs(f'{prefix}/temp')

    # 数据转化
    data_list = list(data for data in os.listdir(prefix) if data.endswith('.dat'))
    index = 0
    for data in data_list:
        label = data.split('.')[0].split('_')[-1]
        Analytical_data(f'{prefix}/{data}', index, 'test', f'{prefix}/temp')
        index += 1
        pass

    # 解析数据
    data_list = list(data for data in os.listdir(f'{prefix}/temp') if data.endswith('.csv'))
    print(f'➡ 待解析文件列表：{data_list}')
    print('++++++++++++++ 解析开始 ++++++++++++++++')

    # 加载模型 mmCountNet
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = mmCountNet().to(device)
    model.load_state_dict(torch.load('models/model.ckpt', map_location=torch.device('cpu')))
    model.eval()

    for data in data_list:
        # 读取数据
        print(f'➡ 当前处理文件为：{data}')
        df_data = pd.read_csv(os.path.join(f'{prefix}/temp', data))

        # 第一轮处理-基础去噪
        df_data = physical_denoise(df_data)
        print('➡ ------数据帧域---------')
        print('➡ ---[起始, 终止, 长度]---')
        df_suspicious_points = find_sff(df_data)
        print('➡ ----------------------')

        # 第二轮处理
        count = 0
        index = 1
        for df_suspicious_point in df_suspicious_points:
            # 轨迹分割
            # snake_list = snake_cluster(df_suspicious_point)

            # 聚类去噪,预测人数
            # print3D(df_suspicious_point)
            # for snake in snake_list:
            #     print3D(snake)
            #     df_data = cluster_denoise(snake)
            #     segment_list = frame_segment(df_data)
            #     density = convert_Numpy(segment_list).reshape(5, 64, 64)
            #     y_pred = predictor(density, model, device)
            #     print(f'➡|   当前轨迹发现{y_pred}人      |')
            #     count += y_pred

            df_data = cluster_denoise(df_suspicious_point)
            segment_list = frame_segment(df_data)
            density = convert_Numpy(segment_list).reshape(5, 64, 64)
            y_pred = predictor(density, model, device)
            print(f'➡|   轨迹{index}发现{y_pred}人        |')
            count += y_pred

            index += 1

        print('➡ ----------------------')
        print(f'➡ 经过分析，数据中的行人个数为{count}人')
        print('+++++++++++++++++++++++++++++++++++++')

    # 清除中间文件
    shutil.rmtree(f'{prefix}/temp')
    pass


def model_evaluate(path, label):
    """
    评价模型的表现
    :param path: 数据路径
    :param label: 数据标签
    :return: 准确率acc
    """
    # 获取数据列表
    data_list = [data for data in os.listdir(path) if data.endswith(f'{label}.csv')]

    # 打乱列表
    shuffle(data_list)

    # 加载模型 mmCountNet
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = mmCountNet().to(device)
    model.load_state_dict(torch.load('models/model.ckpt'))
    model.eval()

    start = time.time()
    # 正确预测的个数
    right_num = 0
    all_num = 0

    for i in range(5):
        # 读取数据
        df_data = pd.read_csv(os.path.join(path, data_list[i]))

        # 数据处理-物理去噪
        df_data = physical_denoise(df_data)

        # 寻找可疑帧域
        df_suspicious_points = find_sff(df_data)

        # 轨迹划分
        for df_suspicious_point in df_suspicious_points:

            snake_list = snake_cluster(df_suspicious_point)
            count = 0
            for snake in snake_list:
                snake = cluster_denoise(snake)
                print(len(snake))
                if len(snake) < 30:
                    continue

                # 帧分片 生成密度图
                segment_list = frame_segment(snake)
                density = convert_Numpy(segment_list)
                # print_density(density)

                # 预测
                y_pred = predictor(density, model, device)
                print(f"第{i + 1}: {data_list[i]}预测结果为{y_pred}")

                all_num += 1
                if y_pred == label:
                    count += 1
                    right_num += 1

                if count > 1:
                    right_num -= 2
                    all_num -= 1
    end = time.time()
    print('time:{}', (end - start) / 5)

    return right_num / all_num


def baseline_evaluate(model='svm'):
    """
    与传统机器学习方案进行对比，包括SVM、决策树
    :param model: svm / dicisionTree
    :return: None
    """

    # 读取数据
    filenames = os.listdir(os.path.join('mmCountData/data_train/seg_csv/'))

    # 合并数据
    X = []
    Y = []
    for file in filenames:
        # 读取数据
        segment = np.array(pd.read_csv(os.path.join('mmCountData/data_train/seg_csv/', file), index_col=0))

        segment = segment[0:64, :].reshape(1, -1)

        # 合成数据和标签
        label = file.split('_')[2][0]
        X.append(segment)
        Y.append(label)

    X = np.array(X).reshape(2000, 4096)
    print('数据读取完成')

    # 划分训练集和测试集
    random_states = [1, 2, 3, 4, 5]
    accuracys = []
    for random in random_states:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random)

        # 创建SVM分类器
        clf = svm.SVC(kernel='linear') if model == 'svm' else DecisionTreeClassifier()

        # 训练模型
        clf.fit(X_train, y_train)

        # 预测测试集
        y_pred = clf.predict(X_test)

        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        accuracys.append(accuracy)
        print(f'Accuracy: {accuracy}')

    print(f'Mean Accuracy : {sum(accuracys) / 5}, Model : {model}')
