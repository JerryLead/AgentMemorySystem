import matplotlib.pyplot as plt

try:
    plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告：可能无法正确显示中文，请安装相应字体")