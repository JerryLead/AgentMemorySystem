import matplotlib.pyplot as plt

try:
    plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.title('测试中文显示')
    plt.xlabel('横坐标')
    plt.ylabel('纵坐标')
    plt.show()
except:
    print("警告：可能无法正确显示中文，请安装相应字体")