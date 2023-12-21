# @ Time : 2022/4/10,11:12
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description: 主函数，给定输入的dat文件 输出行人计数统计

from preprocessor import people_count, model_evaluate

if __name__ == '__main__':

    # 模型测试 测试path路径下文件的准确率
    # path = 'mmCountData/data_csv'
    # acc = []
    # for i in range(5):
    #     acc.append(model_evaluate(path, 4))
    # print(f'5次运行的acc分别为{acc},平均acc为{sum(acc)/5}')

    # 功能函数 预测prefix路径下所有的文件的行人个数
    prefix = 'mmCountData/data_test/scene 4'
    people_count(prefix)

