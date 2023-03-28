pandas采用了大量的NumPy编码风格，但二者最大的不同是pandas是专门为处理表格和混杂数据设计的。而NumPy更适合处理统一的数值数组数据。
## pandas的数据结构介绍
### Series
Series是一种类似于一维数组的对象，它由数据和索引组成。
```python
import pandas as pd
obj = pd.Series([1,3,5,7,8,9])
obj
```
使用values和index属性获取值与索引
```python
In [13]: obj.values
Out[13]: array([ 4,  7, -5,  3])

In [14]: obj.index  # like range(4)
Out[14]: RangeIndex(start=0, stop=4, step=1)
```
自定义索引：
```python
In [15]: obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])

In [16]: obj2
Out[16]: 
d    4
b    7
a   -5
c    3
dtype: int64

In [17]: obj2.index
Out[17]: Index(['d', 'b', 'a', 'c'], dtype='object')
```
获取Series的值：
```python
In [18]: obj2['a']
Out[18]: -5

In [20]: obj2[['c', 'a', 'd']]   
Out[20]:   
c    3
a   -5
d    6
```
保留符合布尔过滤的索引值：
```python
import pandas as pd
obj = pd.Series([1,3,5,7,8,9],index=['a','b','c','d','e','f'])
obj[obj>0]   
``` 
还可以将Series看成是一个定长的有序字典，index就是字典的键，value就是字典的值
```python
obj = pd.Series([1,3,5,7,8,9],index=['a','b','c','d','e','f'])
'a' in obj  # True
```
通过字典的方式来创建Series:
```python
obj3 = pd.Series({'a':-1,'b':3,'c':5})
obj3
```
可以指定自己的index，来获取匹配到的字典，如果没有匹配到则输出NaN
```python
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata,index=states)
obj4
'''
Ohio          35000.0
California        NaN
Oregon        16000.0
Texas         71000.0
dtype: float64
'''
```
检测索引是否匹配到值：
```python
pd.isnull(obj4)
'''
California     True
Ohio          False
Oregon        False
Texas         False
'''
```
根据运算的索引标签自动对齐数据：即会将相同索引值之间进行运算，如果没有相同的操作会变成NaN
```python
obj  = pd.Series([-1,3,5,7,8,9],index=['a','b','c','d','e','f'])
obj3 = pd.Series({'a':-1,'b':3,'c':5})
obj3+obj
'''
a    -2.0
b     6.0
c    10.0
d     NaN
e     NaN
f     NaN
'''
```
## DataFrame
DataFrame是二维的数据结构，进而它既有行索引也有列索引
创建DataFrame：传入字典，只不过字典的值含有多个
```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year':  [2000, 2001, 2002, 2001, 2002, 2003],
        'pop':   [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]
     }
frame = pd.DataFrame(data)
frame
'''
对于结果DataFrame会自动加上索引
	state	year	pop
0	Ohio	2000	1.5
1	Ohio	2001	1.7
2	Ohio	2002	3.6
3	Nevada	2001	2.4
4	Nevada	2002	2.9
5	Nevada	2003	3.2
'''
```
指定原来二维表中列的顺序，以指定的列进行展示：
```python
pd.DataFrame(data, columns=['year', 'state', 'pop'])
'''
   year   state  pop
0  2000    Ohio  1.5
1  2001    Ohio  1.7
2  2002    Ohio  3.6
3  2001  Nevada  2.4
4  2002  Nevada  2.9
5  2003  Nevada  3.2
'''
如果指定的列在数据中找不到，就会在结果中产生缺失值NaN：
```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'])
frame2
# debt 列在原数据中没有，所以展示的时候显示NaN
```
获取某一列的值：
```python
frame2['state']  # 获取这一列中的值
frame2.state     # 使用属性的是方式
```
获取某行数据：
```python
frame2.loc[2]   # 行的索引
```
新增列：
```python
frame1['addtest'] = ['a','d','d','t','e','s']  # 新增一列
```
删除列:
```python
frame1.drop(columns=['addtest'])   # 删除一列
```
删除行：
```python
frame1.drop([1,2])   # 删除指定索引的行
```
> 以上的删除不会对原来的数据产生影响，你输出原来的数据还保持不变，但是当你在删除的时候，加上参数inplace=True就会真正的删除
嵌套字典转二维表：
```python
# 里面的键值作为行索引
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = pd.DataFrame(pop)
frame3
'''
	Nevada	Ohio
2001	2.4	1.7
2002	2.9	3.6
2000	NaN	1.5
'''
```
对二维表进行转置：
```python
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = pd.DataFrame(pop)
frame3.T
'''
	2001	2002	2000
Nevada	2.4	2.9	NaN
Ohio	1.7	3.6	1.5
'''
```
展示指定索引的数据：
```python
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = pd.DataFrame(pop,index=[2000,2001,2002])
frame3
```
## 索引对象
Series获取索引：
```python
 obj = pd.Series(range(3), index=['a', 'b', 'c'])
 index = obj.index
 index
 '''
 Index(['a', 'b', 'c'], dtype='object')
 '''
二维表中获取索引：
```python
frame.columns
Index(['state', 'year', 'pop'], dtype='object')
```
 ## 基本功能
重新指定索引：
 ```python
 # 会根据新的索引进行展示数据，没有该索引的时候展示NaN
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
'''
a   -5.3
b    7.2
c    3.6
d    4.5
e    NaN
'''
```
对于时间序列这样的有序数据，重新索引时可能需要做一些插值处理。method选项即可达到此目的，例如，使用ffill可以实现前向值填充：
```python
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])  # 这个列表只有0，2，4索引值上有数据，当重新索引后，在1，3，5的位置没有值，就会进行填充
obj3.reindex(range(6), method='ffill')
Out[97]:   
0      blue
1      blue
2    purple
3    purple
4    yellow
5    yellow
dtype: object
```
修改二维表的索引：
```python
In [98]: frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
   ....:                      index=['a', 'c', 'd'],
   ....:                      columns=['Ohio', 'Texas', 'California'])

In [99]: frame
Out[99]: 
   Ohio  Texas  California
a     0      1           2
c     3      4           5
d     6      7           8

In [100]: frame2 = frame.reindex(['a', 'b', 'c', 'd'])

In [101]: frame2
Out[101]: 
   Ohio  Texas  California
a   0.0    1.0         2.0
b   NaN    NaN         NaN
c   3.0    4.0         5.0
d   6.0    7.0         8.0
```
修改二维表的列：当指定的列在数据中没有的时候显示NaN
```python
In [102]: states = ['Texas', 'Utah', 'California']

In [103]: frame.reindex(columns=states)
Out[103]: 
   Texas  Utah  California
a      1   NaN           2
c      4   NaN           5
d      7   NaN           8
```
## 索引、选取和过滤
一维数据用切片修改值：
```python
obj['b':'c'] = 5   # 在索引b到c之间的值都会被修改
'''
a    0.0
b    5.0
c    5.0
d    3.0
'''
```
用切片获取二维表中的数据：
```python
data[:2]
'''
          one  two  three  four
Ohio        0    1      2     3
Colorado    4    5      6     7
'''
```
二维表中获取满足条件的行：
```python
data[data['three'] > 5] 
'''
          one  two  three  four
Colorado    4    5      6     7
Utah        8    9     10    11
New York   12   13     14    15
'''
```
## 用loc和iloc进行选取
在二维表中，获取指定行，指定列中的数据：
```python
data.loc['Colorado', ['two', 'three']]   # 获取一行多列的数据
data.iloc[[1, 2], [3, 0, 1]]  # 获取多行多列的数据
data.loc[:'Utah', 'two']   # 使用切片的方式，获取指定的行，指定列的数据
'''
Ohio        0
Colorado    5
Utah        9
'''
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data.iloc[:, :3][data.three > 5]   # 在切片的基础上又增加了bool选择，不显示false的情况
'''
	       one	 two	 three
Colorado	4	  5	   6
Utah	        8	  9	  10
New York        12	 13	  14
'''
```
## 算术运算和数据对齐
1、对于Series相加的话，会将两者的索引取并集，值就是对应的算术运算了
```python
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1],index=['a', 'c', 'e', 'f', 'g'])
s1-s2
'''
a    9.4
c   -6.1
d    NaN
e    3.0
f    NaN
g    NaN
'''
```
对于DataFrame，对齐操作会同时发生在行和列上：
```python
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'), index=['Ohio', 'Texas', 'Colorado'])

df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1 + df2   # 将它们相加时，index与columns会相互对应起来进行算术运算，没有重叠的位置就会产生NaN值
'''
            b   c     d   e
Colorado  NaN NaN   NaN NaN
Ohio      3.0 NaN   6.0 NaN
Oregon    NaN NaN   NaN NaN
Texas     9.0 NaN  12.0 NaN
Utah      NaN NaN   NaN NaN
'''
```
> 相加产生新的二维表的index与columns会排好序进行展示
## 在算术方法中填充值
使用df1的add方法，传入df2以及一个fill_value参数：
fill_value参数会将df3中的值为NaN的时候，设置为这个填充的值，然后进行算术操作
```python
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
                 columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
                 columns=list('abcde'))
df3 = df1+df2
print(df2)
df2.add(df3,fill_value = 0)
```
- sub 用于减法
- div 用于除法
- floordiv 用于底除
- mul 用于乘法
- pow 用于指数
## DataFrame和Series之间的运算
计算二维数组与它的某行之间的差：
```python
arr = np.arange(12.).reshape((3, 4))
'''
array([[  0.,   1.,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.]])
'''
arr[0]
'''
array([ 0.,  1.,  2.,  3.])
'''
arr - arr[0]
'''
array([[ 0.,  0.,  0.,  0.],
       [ 4.,  4.,  4.,  4.],
       [ 8.,  8.,  8.,  8.]])
'''
```
> 当我们从arr减去`arr[0]`，每一行都会执行这个操作。这就叫做广播

DataFrame和Series之间的算术运算会**沿着行一直向下广播**:
```python
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                    columns=list('bde'),
                    index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]
frame - series 
'''
	b	d	e
Utah	0.0	0.0	0.0
Ohio	3.0	3.0	3.0
Texas	6.0	6.0	6.0
Oregon	9.0	9.0	9.0
'''
```
算术操作会列举出所有的索引值： 当有一个表中没有这个索引的时候显示NaN
```python
series2 = pd.Series(range(3), index=['b', 'e', 'f'])
frame + series2
'''
         b   d     e   f
Utah    0.0 NaN   3.0 NaN
Ohio    3.0 NaN   6.0 NaN
Texas   6.0 NaN   9.0 NaN
Oregon  9.0 NaN  12.0 NaN
'''
```
对二维表在columns中进行算术运算，需要采用二维表中的算术方法进行操作。例如：让二维表减去某一列的值
```python
series3 = frame['d']
print(frame)
'''
          b     d     e
Utah    0.0   1.0   2.0
Ohio    3.0   4.0   5.0
Texas   6.0   7.0   8.0
Oregon  9.0  10.0  11.0
'''
print(series3)
'''
Utah       1.0
Ohio       4.0
Texas      7.0
Oregon    10.0
'''
frame.sub(series3, axis=0)  # 指明在列上进行广播
```
## 函数应用和映射
numpy中的函数操作pandas对象：
```python
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),index=['Utah', 'Ohio', 'Texas', 'Oregon'])
np.abs(frame)
```
计算每一列中的最大值与最小值的差：
```python
f = lambda x: x.max() - x.min()
frame.apply(f,axis= 0)
'''
b    1.802165
d    1.684034
e    2.689627
'''
```
计算每一行中的最大值与最小值的差：
```python
frame.apply(f, axis=1)
'''
Utah      0.998382
Ohio      2.521511
Texas     0.676115
Oregon    2.542656
'''
```
注意：常见的数组统计功能都被实现成DataFrame实现了，直接调用这个对象的方法即可
## 排序和排名
### 按照索引排序
对一维的Series利用索引进行排序：
```python
obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()
'''
a    1
b    2
c    3
d    0
'''
```
对二维表的数据对列的方向进行排序：
```python
frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                 index=['three', 'one'],
                 columns=['d', 'a', 'b', 'c'])
frame.sort_index(axis=0)
'''
       d  a  b  c
one    4  5  6  7
three  0  1  2  3
'''
```
对二维表的数据对行的方向进行排序：
```python
frame.sort_index(axis=1)
'''
       a  b  c  d
three  1  2  3  0
one    5  6  7  4
'''
```
数据默认是按升序排序的，如果进行降序排序，则:frame.sort_index(axis=1, ascending=False)
### 按值排序
对于一维的Series列表：
```python
obj = pd.Series([4, 7, -3, 2])
obj.sort_values()
'''
2   -3
3    2
0    4
1    7
'''
```
对于二维表的数据，你需要指定对一列或者多列的值进行排序：
```python
frame = pd.DataFrame(np.arange(8).reshape((2, 4)),index=['three', 'one'],columns=['d', 'a', 'b', 'c'])
frame.sort_values(by='a')
```
## 带有重复标签的轴索引
虽然许多pandas函数（如reindex）都要求标签唯一，但这并不是强制性的。我们来看看下面这个简单的带有重复索引值的Series：
```python
obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
```
索引的is_unique属性可以告诉你它的值是否是唯一的：obj.index.is_unique  False
如果某个索引对应多个值，则返回一个Series；而对应单个值的，则返回一个标量值：
```python
obj['a']
'''
a    0
a    1
'''
obj['c']
'''
4
'''
```
## 汇总和计算描述统计
pandas对象拥有一组常用的数学和统计方法。它们大部分都属于约简和汇总统计，用于从Series中提取单个值（如sum或mean）或从DataFrame的行或列中提取一个Series。跟对应的NumPy数组方法相比，它们都是基于没有缺失数据的假设而构建的。看一个简单的DataFrame：
调用DataFrame的sum方法将会返回一个含有列的和的Series：
```python
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                [np.nan, np.nan], [0.75, -1.3]],
                index=['a', 'b', 'c', 'd'],
                 columns=['one', 'two'])
'''
    one  two
a  1.40  NaN
b  7.10 -4.5
c   NaN  NaN
d  0.75 -1.3
'''
df.sum()
'''
one    9.25
two   -5.80
'''
```
传入axis=1将会按行进行求和运算：
```python
df.sum(axis=1)
'''
a    1.40
b    2.60
c     NaN
d   -0.55
'''
```
NA值会自动被排除，除非整个行或列都是NA。通过skipna选项可以排除这种机制：即只要是NaN相加就会直接是NaN
```python
df.mean(axis='columns', skipna=False)
'''
a      NaN
b    1.300
c      NaN
d   -0.275
'''
```
**约简方法的常用选项**
- axis 约简的轴。DataFrame的行用的0，列用的1
- skipna 排除缺失值，默认值为True
- level 如果轴是层次化索引，则根据level分组约简
有些方法（如idxmin和idxmax）返回的是间接统计（比如达到最小值或最大值的索引）：
```python
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                [np.nan, np.nan], [0.75, -1.3]],
                index=['a', 'b', 'c', 'd'],
                 columns=['one', 'two'])
print(df)
df.idxmax()
'''
    one  two
a  1.40  NaN
b  7.10 -4.5
c   NaN  NaN
d  0.75 -1.3
one    b
two    d
dtype: object
'''
```
另一些方法则是累计型的：
```python
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                [np.nan, np.nan], [0.75, -1.3]],
                index=['a', 'b', 'c', 'd'],
                 columns=['one', 'two'])
print(df)
df.cumsum()
'''
    one  two
a  1.40  NaN
b  7.10 -4.5
c   NaN  NaN
d  0.75 -1.3
one	two
a	1.40	NaN
b	8.50	-4.5
c	NaN	NaN
d	9.25	-5.8
'''
```
还有一种方法，它既不是约简型也不是累计型。describe就是一个例子，它用于一次性产生多个汇总统计：
```python
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                [np.nan, np.nan], [0.75, -1.3]],
                index=['a', 'b', 'c', 'd'],
                 columns=['one', 'two'])
print(df)
df.describe()
'''
    one  two
a  1.40  NaN
b  7.10 -4.5
c   NaN  NaN
d  0.75 -1.3
one	two
count	3.000000	2.000000
mean	3.083333	-2.900000
std	3.493685	2.262742
min	0.750000	-4.500000
25%	1.075000	-3.700000
50%	1.400000	-2.900000
75%	4.250000	-2.100000
max	7.100000	-1.300000
'''
```
**统计相关的方法**
![](http://upload-images.jianshu.io/upload_images/7178691-11fa967f658ac314.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
## 唯一值、值计数以及成员资格
还有一类方法可以从一维Series的值中抽取信息。看下面的例子：
```python
obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
```
第一个函数是unique，它可以得到Series中的唯一值数组：
```python
uniques = obj.unique()
```
返回的唯一值是未排序的，如果需要的话，可以对结果再次进行排序（uniques.sort()）。相似的，value_counts用于计算一个Series中各值出现的频率：
```python
obj.value_counts()
```
isin用于判断矢量化集合的成员资格，可用于过滤Series中或DataFrame列中数据的子集：
```python
mask = obj.isin(['b', 'c'])
'''
0     True
1    False
2    False
3    False
4    False
5     True
6     True
7     True
8     True
'''
obj[mask]
'''
0    c
5    b
6    b
7    c
8    c
'''
```
