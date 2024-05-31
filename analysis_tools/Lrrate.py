import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
# 文件路径
file_path = '20240414_221012.log'

# 读取文件内容
with open(file_path, 'r') as file:
    log_content = file.readlines()

# 提取包含学习率的行
lr_lines = [line for line in log_content if 'lr' in line and 'Iter(train)' in line]

# 初始化列表存储迭代次数和学习率
iterations = []
lrs = []

# 解析数据
for line in lr_lines:
    parts = line.split()
    # 提取迭代次数和学习率
    iter_part = ' '.join(parts)  # 为处理空格，合并行元素
    iter_num = int(iter_part.split('Iter(train) [')[1].split(']')[0].split('/')[0].strip())
    lr = float(iter_part.split('lr:')[1].split()[0].strip())

    iterations.append(iter_num)
    lrs.append(lr)

# 绘图
plt.figure(figsize=(10,8))
plt.plot(iterations, lrs, marker='o', linestyle='-')
# plt.title('Learning Rate Changes During Training')
# plt.xlabel('Iteration Number', fontsize=20)
# plt.ylabel('Learning Rate', fontsize=20)
plt.ylim(0, max(lrs) * 1.1)  # 调整y轴范围以更好地显示小值
# plt.grid(True)
#
import matplotlib as mpl
mpl.rcParams['axes.formatter.limits'] = (-1, 1)
mpl.rcParams['xtick.labelsize'] = 20  # Adjust x-axis tick label size
mpl.rcParams['ytick.labelsize'] = 20  # Adjust y-axis tick label size
mpl.rcParams['axes.labelsize'] = 20   # Adjust axis label size
mpl.rcParams['mathtext.fontset'] = 'stix'  # Optional: change to a different math font style
mpl.rcParams['font.size'] = 30
# x_formatter = ScalarFormatter(useMathText=True)
# x_formatter.set_scientific(True)
# x_formatter.set_powerlimits((-1, 1))  # Adjust as necessary for your data range
# plt.gca().xaxis.set_major_formatter(x_formatter)

# Formatting y-axis for clarity, optional
y_formatter = ScalarFormatter(useMathText=True)
y_formatter.set_scientific(True)
y_formatter.set_powerlimits((-1, 1))
plt.gca().yaxis.set_major_formatter(y_formatter)

plt.tick_params(axis='both', which='major', labelsize=24)

plt.gca().yaxis.get_offset_text().set_fontsize(20)
plt.gca().yaxis.get_offset_text().set_verticalalignment('bottom')
plt.savefig('pl.tif', dpi=120)
plt.show()
