## ndarray 多维数组对象
ndarray是python中一个快速、灵活的大型数据集容器，它使用类似于Python内建对象的标量计算语法进行批量计算，我首先导入NumPy,再生成一个小的随机数组，让你领略numpy如何进行批量计算的:
```python
import numpy as np
data = np.random.randn(2, 3)
data*10
```
结果：
array([[ 12.99469322, -12.05703843,   6.9673719 ],
       [ -0.59881384, -10.60593603,  10.68280631]])
> 说明：randn函数是生成一个标准正态分布的函数       
       
ndarray它包含的每一个元素都必须是相同的数据类型。每一个数组都有一个shape和dtype属性，可以得到数组的形状和里面存储的数据类型
```python
data.shape   # 数组的形状
data.dtype   # 数组的数据类型
```
> 说明：当你看到“数组”、“NumPy数组”或“ndarray”时，他们都表示同一个对象: ndarray 对象。

## 生成ndarry数组的方式
生成数组最简单的方式就是使用array函数。它接受一切序列型的对象（包括其他数组），然后产生一个新的含有传入数据的NumPy数组。例如，列表的转换
```python
data1 = [6, 7.5, 8, 0, 1]
np.array(data1)
```
在嵌套序列中，例如同等长度的列表，将会自动转换成多维数组:  注意如果不是同等长度的不能转换
```python
data1 = [[6, 7.5, 8], [0, 1,3]]
np.array(data1)  # 生成二维数组
```
## 其他函数创建数组
1. 给定长度与形状后，zeros可以一次性创造全0的数组：np.zeros(3) 
2. zeros_like根据所给数组生成一个形状一样的全0数组
3. 给定长度与形状后，ones可以一次性创造全1的数组：np.ones(3)  
4. ones_like根据所给的数组生成一个形状一样的全1数组
5. 想要创建高维的数据，需要给shape传递一个元组：np.zeros((2,3))  形成二维的数组 
6. 使用arange函数：np.arange(10)   **它是内建函数range的数组版，即会产生一维的数组**

## ndarray的数据类型
数据类型，即dytpe,d是date的缩写，中文就是数据类型的意思，它包含了ndarray需要为某一种类型数据所申明的内存块信息(也称为元数据，即表示数据的数据):
```python
arr1 = np.array([1, 2, 3], dtype=np.float64) 
arr1.dtype     # dtype( ' float64' )
arr2 = np.array([1, 2, 3], dtype=np.int32)      
arr2.dtype     # dtype( ' int32' )
```

我们可以使用astype方法转换数组的数据类型:
```python
arr = np.array([1, 2, 3, 4, 5])
float arr = arr.astype(np.float64)  # 将上面的整数类型转为浮点数

arr = np.array([3.7,-1.2, -2.6, 0.5, 12.9, 10.1] )
arr = arr.astype(np. int32)
arr  # 会将小数部分截取
```
如果你有一个数组，里面的元素都是表达数字含义的字符串，也可以通过astype将字符串转换为数字:
```python
numeric_strings = np.array(['1.25', '-9.6','42'], dtype=np.string_)
numeric_strings.astype(float )
```
注意：在使用astype时，总会生成一个新的数组。即使你在astype中传入的dtype与之前一样
## NumPy数组算术

数组之所以重要是因为它**允许你进行批量操作而无须任何for循环。NumPy用户称这种特性为向量化**。任何在两个等尺寸数组之间的算术操作都应用了逐元素操作的方式:
```python
arr = np.array([[1., 2. , 3.], [4., 5., 6.]])
arr+1
```
会进行逐元素相加，带有标量计算的算术操作，会把计算参数传递给数组的每一个元素

同尺寸数组之间的比较，会产生一个布尔值数组:
```python
arr = np.array([[1., 2. , 3.], [4., 5., 6.]])
arr1 = arr+1
arr1>arr

'''array([[ True,  True,  True],
       [ True,  True,  True]])
'''       
```
注意：不同尺寸的数组间的操作，将会用到广播特性，将会在附录A中介绍。对于本书大部分内容来说，并不需要深入理解广播特性。
## 基础索引与切片
### 一维数组比较简单，看起来和Python的列表很类似:
```python
arr = np.arange(10)
arr[1]
arr[1:5]
arr[3] = 8
```
区别于Python的内建列表，数组的切片是原数组的视图。这意味着数据并不是被复制了，任何对于视图的修改都会反映到原数组上：
```python
arr = np.arange(10)
arr[3:8] = 8
arr
# out  array([0, 1, 2, 8, 8, 8, 8, 8, 8, 9])
```
即当你对切片操作进行修改的时候，也会对原数组进行修改

如果你还是想要一份数组切片的拷贝而不是一份视图的话，你就必须显式地复制这个数组，例如`arr[5:8].copy()`。视图的意思就是你获得数据就是原来的数组本身，你修改试图，本体也会发生改变
### 二维数组中
在一个二维数组中，每个索引值对应的元素不再是一个值，而是一个一维数组: 
```python
arr2d = np.array([
    [1, 2,3], 
    [4, 5,6], 
    [7, 8,9]
])
arr2d[1]
```
### 在三维数组中
在多维数组中，你可以省略后续索引值，返回的对象将是降低一个维度的数组。因此在一个2X2X3的数组arr3d中:
```python
arr3d = np.array([
                  [[1,2,3],[4,5,6]],
                  [[7,8,9],[10,11,12]]
                 ])
arr3d[1,1,0]   # 表示得到第一行，第一列中的索引为1的值
```
> 怎么区分二维、三维数组？我们看中括号，数几个中括号遇到值后，就表示的是几维数组
## 二维数组的切片索引
```python
arr2d = np.array([[1, 2,3], [4, 5, 6], [7, 8,9]])
print(arr2d[0:2])  # arr2d[:2]  开头与结尾的数据可以省略不写，没有逗号的就是仅仅表示行
print(arr2d[2:3])  # arr2d[2:]
```
 ### 进行多组切片
 ```python
arr2d = np.array([[1, 2,3], [4, 5, 6], [7, 8,9]])
print(arr2d[:2,:])  # 表示取0到1行中的所有的列
```
**当列表中有逗号分隔，表示前面拿的是行的数据，后面拿的是列的数据**
## 布尔索引
### 一维数组的索引
布尔数组中，下标为0,3,4的位置是True，**会取对应位置的元素**。
```python
    In [24]: arr = np.arange(7)
    In [25]: booling1 = np.array([True,False,False,True,True,False,False])
    In [26]: arr[booling1]
    Out[26]: array([0, 3, 4])
 ```
### 二维数组的索引
布尔数组中，下标为0,3,4的位置是True，因此将**会取**出目标数组中第0,3,4**行**。
```python     
 In [27]: arr = np.arange(28).reshape((7,4))
    
    In [28]: arr
    Out[28]: 
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27]])
    
    In [29]: booling1 = np.array([True,False,False,True,True,False,False])
    
    In [30]: arr[booling1]
    Out[30]: 
    array([[ 0,  1,  2,  3],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
``` 
假设我们的数据都在数组中，并且数组中的数据是一些存在重复的人名。我会使用numpy.random中的randn函数来生成一些随机正态分布的数据:
```python
import numpy as np
names = np.array(['abc','b','x','abc','abc'])
names  
# out  array(['abc', 'b', 'x', 'abc', 'abc'], dtype='<U3')   
# 所以制作的二维数组的横轴也要为3
```
假设每个人名都和data数组中的一行相对应，所以要形成5行，而列的个数与字符串的最大长度一样
我们需要选中所有'abc'对象的行:
```python
import numpy as np
names = np.array(['abc','b','x','abc','abc'])
data = np.random.randn(5,3)
data[names == 'abc']   # 本质上还是与切片是一样的，在方括号中的第一个参数是选着的行的方向
```
分析：
names == 'abc'的结果是一个布尔类型的数组 `[ True, False, False,  True,  True]`，我们将这个布尔类型的数组传给一个数组
它会进行匹配，每一个布尔值对应着这个数组的一行数据，所以布尔数组的维度要与匹配数组的行的维度一样，最后就会返回对应值为True的行，形成一个数组返回
`array([[-0.41402997,  0.27142868, -0.10981186],
       [ 0.83403482, -0.1846209 , -2.07761868],
       [-0.17093358, -1.29584302, -1.38154292]])`

选择 names == Bob'的行，并索引了各个列:
```python
data[names == 'abc',2:]  # 第一个参数是选着行后，第二个参数继续进行筛选
```
为了选择除了'abc'以外的其他数据，你可以使用!=或在条件表达式前使用~对条件取反: 
`data[names != 'abc']`  或者 `data[~(names== 'abc')]`

当要选择三个名字中的两个时，可以对多个布尔值条件进行联合，需要使用数学操作符如& (and) 和| (or):
```python
mask ==( names =='abc')|(name == 'b')
data[mask]
```
**使用布尔值索引选择数据时，总是生成数据的拷贝，即使返回的数组并没有任何变化。**
> 说明：Python的关键字and和or对布尔值数组并没有用，请使用& (and) 和|(or)来代替。
### 基于常识来设置布尔值数组的值
将data中所有的负值设置为0,我们需要做:
```python
import numpy as np
data = np.random.randn(2,3)
print(data <0)
data[data < 0 ] = 0
data
``` 
## 数组转置与换轴
转置：用数组transpose方法，或者特殊的T属性:
```python
arr= np. arange(15) .reshape((3, 5))
arr.T
```
矩阵的内积：
两个向量对应分量乘积之和。np.dot(A,B)
```python
arr= np. arange(15) .reshape((3, 5))
A = arr.transpose()
np.dot(arr,A)
```
结果：`array([[ 30,  80, 130],
       [ 80, 255, 430],
       [130, 430, 730]])`
## 通用函数[ufunc]，逐元素处理函数
比如一元通用函数：sqrt或exp函数:
```python
arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)
```
比如二元通用函数：add或maximum函数
```python
x = np.random.randn(8)
y = np.random.randn(8)
np.maximum(x,y)
```
这里，numpy.maximum逐个元素地将x和y中元素的最大值计算出来。
## 一元通用函数
- abs、fabs   逐个元素计算整数、浮点数或复数的绝对值
- sqrt 计算每个元素的平方根
- square  计算每个元素的平方
- exp 计算每个元素的自然指数值e^x
- log log10 log2 log1p   分别对应自然对数（e为底）、对数10为底、对数2为底、log(1+x)
- sign  计算每个元素的符号值：1（正数）、0（0）、-1（负数）
- ceil  计算每个元素的最高整数值（即大于等于给定值的最小整数）
- floor 计算每个元素的最小整数值（即小于等于给定元素的最大整数）
- rint  将元素保留到整数位，并保持dtype
- modf 分别将数组的小数部分和整数部分按数组形式返回
- isnan 返回数组中的元素是否是一个NaN(不是一个数值)，形式为布尔值数组
- isfinite、isinf   分别返回数组中的元素是否有限（非inf、非NaN）、是否无限的，形式为布尔值数组
## 一元通用函数
- add 将数组对应的元素相加
- subtract 在第二个数组中，将第一个数组中包含的元素去除
- multiply 将数组对应元素相乘
- divide，floor_divide  除或整除（放弃余数）
- power将第二个数组的元素作为第一个数组对应元素的幂次方
- maximum，fmax 逐个元素计算最大值，fmax忽略NaN
- minimun，fmin 逐个元素计算最小值，fmin忽略NaN
- mod 按元素的求模计算（即求除法的余数）  
## 将条件逻辑作为数组操作
假设我们有一个布尔数组和两个值数组：
```python
In [165]: xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])

In [166]: yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])

In [167]: cond = np.array([True, False, True, True, False])
```
假设我们想当cond中的值为True时，选取xarr的值，否则从yarr中选取
```python
In [170]: result = np.where(cond, xarr, yarr)

In [171]: result
Out[171]: array([ 1.1,  2.2,  1.3,  1.4,  2.5])
```
> where函数的典型用法就是根据一个数组来生成一个新数组
假如你有一个随机生成的数组，你想将大于0的数，改为2，将小于0的数改为1
```python
arr = np.random.randn(2,3)
np.where(arr>0,2,1)
```
## numpy中的数学与统计方法
- sum 沿着轴向计算所有元素的累和，0长度的数组，累和为0
- mean 数据平均，0长度的数组平均值为NaN
- std、var 标准差和方差，可以选择自由度调整
- min，max 最小值与最大值
- argmin，argmax 最小值和最大值的位置
- cumsum  所有元素累和
- cumprod 所有元素累积

mean和sum这类的函数可以接受一个axis选项参数，用于计算该轴向上的统计值，最终结果是一个少一维的数组：
```python
arr = np.random.randn(5, 4)
np.mean(arr, axis=0)

[[ 0.33379297 -1.20234126  0.92199071  1.2139666 ]
 [ 0.29498613 -0.32915131  0.49206796  2.99880893]
 [ 0.9085492   0.69002296 -1.40754324  1.3563622 ]
 [-0.16883191  0.49922795 -0.35523631  1.39041476]
 [ 1.33967881 -1.88807552  0.08659239 -1.25322137]]
 
array([-0.3752747 ,  0.00285498,  0.36041613, -0.21401022])  # 以竖轴方向上做的平均值

```
## 布尔数组中的方法
如果我们想统计出布尔数组中True的个数，那么可以这样做：
```python
arr = np.array([True,False,True])
np.sum(arr)
```
any用于测试布尔数组中是否存在一个或多个True，而all则检查布尔数组中所有值是否都是True：
```python
bools = np.array([False, False, True, False])
bools.any()  #  是否存在True
bools.all()  #  是否都是True
```
这两个方法也能用于非布尔型数组，所有非0元素将会被当做True。
## numpy排序
numpy数组方法直接有一个排序的方法，当然了numpy命名空间中也有一个排序方法
```python
xarr = np.array([4, 1.2, 1.3, 1.4, 1.5])
xarr.sort()   # 这个排序后不能给它赋值
xarr
```
对于多维数组可以指定在一个轴上排序，只需将轴编号传给sort
```python
arr = np.random.randn(5, 3)
arr.sort(1)
arr
```
## 数组的集合操作
- unique(x)   计算x的唯一值，并排序
- intersect1d(x,y)  计算x和y的交集，并排序
- union1d(x,y)     计算x和y的并集，并排序
- in1d(x,y)        计算x中的元素是否包含在y中，返回一个布尔值数组
- setdiff1d(x,y)   差集，在x中但不在y中的元素
- setxor1d(x,y)    异或集，在x或y中，但不属于x，y交集的元素
## numpy中的线性代数操作
- diag 将一个方阵的对角（或非对角）元素作为一维数组返回，或者将一维数组转换成一个方阵，并且在非对角线上补0
- dot  矩阵的点乘
- trace 计算矩阵的迹
- det 计算矩阵的行列式
- eig 计算方阵的特征值和特征向量
- inv 计算方阵的逆矩阵
- pinv 计算矩阵的Moore-Penrose伪逆【广义逆】
- qr 计算QR分解
- svd 计算奇异值分解（SVD）
- solve  求解x的线性系统Ax=b，其中A是方阵
- lstsq  计算 Ax=b 的最小二乘解

## 伪随机数
numpy.random模块是对Python内置的random进行了补充，内置的random模块则只能一次生成一个样本值，而使用numpy中的random模块可以一次性生成指定数组样式的随机数。我们说这些都是伪随机数，因为它们都是通过算法基于随机数生成器种子，在确定性的条件下生成的。每一次运行随机数的种子都不一样，是的产生的结果也就不一样，如果想将得到的伪随机数都是一样的，则需要指定随机数种子`np.random.seed(0)`

**根据给定的序列产生该序列的随机生成序列**
- np.random.shuffle(arr) 对一个序列就地随机排列
- np.random.permutation(arr)  返回一个序列的随机排序值  

- np.random.beta() 产生Beta分布的样本值


- np.random.randint(low = 1, high=10, size=(3,3), dtype=int)   从给定的上下限里生成随机整数组成的数组，例如这里`[1,10)`，size指定生成的数组尺寸
- np.random.random(size=(2,2)) 生成`[0., 1.)`之间均匀分布的随机数组，size为生成的尺寸
- numpy.random.rand(d0,d1,…,dn)   `生成[0,1)之间的数据`  dn表格每个维度，例如rand(4,3,2) # shape: 4*3*2
- numpy.random.randn(d0,d1,…,dn)   randn函数返回一个或一组样本，具有标准正态分布。【标准正态分布又称为u分布，是以0为均值、以1为标准差的正态分布，记为N（0，1）】

**产生正态分布的样本值**
- numpy.random.normal(loc=0,scale=1e-2,size=shape) 
参数loc(float)：正态分布的均值，对应着这个分布的中心。loc=0说明这一个以Y轴为对称轴的正态分布，
参数scale(float)：正态分布的标准差，对应分布的宽度，scale越大，正态分布的曲线越矮胖，scale越小，曲线越高瘦。
参数size(int 或者整数元组)：输出的值赋在shape里，默认为None。

**产生二项分布的样本值**
二项分布是由伯努利提出的概念，指的是重复n次（注意：这里的n和binomial()函数参数n不是一个意思）独立的伯努利试验，如果事件X服从二项式分布，则可以表示为X~B(n,p)，则期望E(X)=np，方差D(X)=np(1-p)。简单来讲就是在每次试验中只有两种可能的结果（例如：抛一枚硬币，不是正面就是反面，而掷六面体色子就不是二项式分布），而且两种结果发生与否互相对立，并且相互独立，与其它各次试验结果无关，事件发生与否的概率在每一次独立试验中都保持不变。
- numpy.random.binomial(n,p,size=None)
参数n：一次试验的样本数n
参数p：事件发生的概率p，`范围[0,1]`
size是一个整数N时，返回一个长度为N的一维数组；size是（X，Y）类型元组时，返回一个X行Y列二维数组；size是（X，Y，Z）类型元组时，返回一个三维数组








