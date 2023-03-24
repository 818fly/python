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
       
ndarray它包含的每一个元素都必须是相同的数据类型。每一个数组都有一个shape和dtype属性，可以得到数组的形状和里面存储的数据类型
```python
data.shape   # 数组的形状
data.dtype   # 数组的数据类型
```
说明：当你看到“数组”、“NumPy数组”或“ndarray”时，他们都表示同一个对象: ndarray 对象。

## 生成ndarry数组的方式
生成数组最简单的方式就是使用array函数。例如，列表的转换
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
6. 使用arange函数：np.arange(10)   它是内建函数range的数组版，即会产生一维的数组

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
arr2d = np.array([[1, 2,3], [4, 5, 6], [7, 8,9]])
arr2d[0]
# out array([1, 2, 3])
```
因此，单个元素可以传递一个索引的逗号分隔列表去选择单个元素，以下两种方式效果一样:
```python
arr2d[0,2]  # 这种方式比较好，我认为更符合python语言，逗号前面的表示行，逗号后面的表示列
arr2d[0][2]
```
### 在三维数组中
在多维数组中，你可以省略后续索引值，返回的对象将是降低一个维度的数组。因此在一个2X2X3的数组arr3d中:
```python
arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
arr3d[0]   #  一个2x3的数组
arr3d[1][0]   # 一个一维的数组
```
## 二维数组的切片索引
```python
arr2d = np.array([[1, 2,3], [4, 5, 6], [7, 8,9]])
print(arr2d[0:2])  # arr2d[:2]  开头与结尾的数据可以省略不写，没有逗号的就是仅仅表示行
print(arr2d[2:3])  # arr2d[2:]
```
**拿到的是行的数据** ，分别为：` [[1 2 3]
 [4 5 6]]`
 `[[7 8 9]]`
 ### 进行多组切片
 ```python
arr2d = np.array([[1, 2,3], [4, 5, 6], [7, 8,9]])
print(arr2d[:2,:])
```
**当列表中有逗号分隔，表示前面拿的是行的数据，后面拿的是列的数据**
结果：`[[1 2 3]
 [4 5 6]]`
 ### 将索引与切片混合
选择第二行但是只选择前两列:
```python
import numpy as np
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d[1,:2])
# out [4 5]
```
选择第三列，但是只选择前两行:
```python
import numpy as np
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d[0:2,2])
# out [3,6]    
```
当然你对切片表达式进行赋值时，整个切片都会重新赋值
## 布尔索引
### 一维数组的索引

布尔数组中，下标为0,3,4的位置是True，因此将会取出目标数组中对应位置的元素。
```python
    In [24]: arr = np.arange(7)
    
    In [25]: booling1 = np.array([True,False,False,True,True,False,False])
    
    In [26]: arr[booling1]
    Out[26]: array([0, 3, 4])
 ```

### 二维数组的索引
布尔数组中，下标为0,3,4的位置是True，因此将会取出目标数组中第0,3,4行。
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
注意：Python的关键字and和or对布尔值数组并没有用，请使用& (and) 和|(or)来代替。
### 基于常识来设置布尔值数组的值
将data中所有的负值设置为0,我们需要做:
```python
import numpy as np
data = np.random.randn(2,3)
print(data <0)
data[data < 0 ] = 0
data
```
是什么意思呢？应该就是数组中的每一个值形成布尔，然后也会将满足这个布尔值的赋值相应的元素
说明：这里的print(data)的数据是：
`[[False  True False]
 [False  True False]]`
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
## 通用函数:快速的逐元素数组函数
通用函数是ndarray数组进行逐元素操作的函数
比如一元通用函数：sqrt或exp函数:
```python
arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)
```
比如add或maximum则会接收两个数组并返回一个数组作为结果的二元通用函数
```python
x = np.random.randn(8)
y = np.random.randn(8)
print(x)
print(y)
np.maximum(x,y)
```
这里，numpy.maximum逐个元素地将x和y中元素的最大值计算出来。
当然还有很多的通用函数，具体见《利用python进行数据分析》108页


