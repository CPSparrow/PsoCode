# PSO 代码说明

## Usage

1. ```func_plot.py```:绘制了给定区域内的函数图像。修改x、y的范围为[0,10)就可以看到函数在搜索空间内的图像。
2. ```pso.py```:pso算法的核心部分。直接运行该代码的话会运行一次pso算法，输出最佳的函数值以及对应的X、Y坐标。此时速度以标准正态分布初始化并使用cosine递减。
3. ```mulit_test.py```:默认运行200次pso并统计数据。复现不同设置下的算法可以选择修改 ```multi_test``` 函数中的 ```schedule_type``` 和 ```init_type```这两个参数。
4. 更详细的说明：
    1. 修改 $\omega_{min}$ 需要手动在 ```pso.py``` 的```schedule``` 函数中修改
   2. ```schedule_type```：控制惯性参数的变化方式，接受3种参数：
       - ```None```:惯性参数不变
       - ```"linear"```:线性递减
       - ```"cos"```:余弦递减
   3. ```init_type```:控制粒子的初始化方式，接受4类参数：
       - ```None```:标准正态初始化
       - ```"zero"```:初始化为0
       - ```"uniform"```:在[-1,1]范围内随机均匀初始化
       - ```float```:接受一个浮点数，控制正态初始化的标准差

## Installation
代码下载：
```bash
git clone https://github.com/CPSparrow/pso.git
```
依赖安装：
```bash
pip install numpy tqdm
```
