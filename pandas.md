pandas采用了大量的NumPy编码风格，但二者最大的不同是pandas是专门为处理表格和混杂数据设计的。而NumPy更适合处理统一的数值数组数据。
## pandas的数据结构介绍
### Series
Series是一种类似于一维数组的对象，它由数据和索引组成。
```python
import pandas as pd
obj = pd.Series([1,3,5,7,8,9])
obj
```
你可以通过Series 的values和index属性获取其数组表示形式和索引对象：
```python
In [13]: obj.values
Out[13]: array([ 4,  7, -5,  3])

In [14]: obj.index  # like range(4)
Out[14]: RangeIndex(start=0, stop=4, step=1)
```
通常，我们希望所创建的Series带有一个可以对各个数据点进行标记的索引：
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
与普通NumPy数组相比，你可以通过索引的方式选取Series中的单个或一组值：
```python
In [18]: obj2['a']
Out[18]: -5

In [19]: obj2['d'] = 6    # 修改值

In [20]: obj2[['c', 'a', 'd']]   #['c', 'a', 'd']是索引列表
Out[20]:   
c    3
a   -5
d    6
dtype: int64
```
使用NumPy函数或类似NumPy的运算（如根据布尔型数组进行过滤、标量乘法、应用数学函数等）都会保留对应的索引值链接：
```python
import pandas as pd
import numpy as np
obj = pd.Series([1,3,5,7,8,9],index=['a','b','c','d','e','f'])
obj[obj>0]    # 会保留符合布尔过滤的索引值
np.exp(obj)   # 使用np中的exp函数后，传入的是pd对象，也会得到对应值得索引
``` 
还可以将Series看成是一个定长的有序字典，因为它是索引值到数据值的一个映射。它可以用在许多原本需要字典参数的函数中：
```python
obj = pd.Series([1,3,5,7,8,9],index=['a','b','c','d','e','f'])
'a' in obj  # True
```
**如果数据被存放在一个Python字典中，也可以直接通过这个字典来创建Series：**
```python
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}   #  直接满足了索引与值得要求
obj3 = pd.Series(sdata)
obj3
```
你可以传入排好序的字典的键以改变原来字典的顺序：
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
> sdata中跟states索引相匹配的那3个值会被找出来并放到相应的位置上，但由于"California"所对应的sdata值找不到，所以其结果就为NaN。因为‘Utah’不在states中，它被从结果中除去。

pandas的isnull和notnull函数可用于检测有索引但是缺失数据的情况：
```python
pd.isnull(obj4)
'''
California     True
Ohio          False
Oregon        False
Texas         False
dtype: bool
'''
```
对于许多应用而言，Series最重要的一个功能是，它会根据运算的索引标签自动对齐数据：
例如：将上面的pd形成的ojb3和obj4对象相加
```python
obj3 + obj4
'''
    California         NaN
    Ohio           70000.0
    Oregon         32000.0
    Texas         142000.0
    Utah               NaN
'''
```
数据对齐功能就类似与数据库中的join的操作，但是又不一样
## DataFrame
DataFrame是一个表格型的数据结构，DataFrame中的数据是以一个或多个二维块存放的（而不是列表、字典或别的一维数据结构）。DataFrame既有行索引也有列索引
建DataFrame的办法有很多，最常用的一种方式是传入NumPy数组组成的字典：  将会把数据纵列显示，其中字典的键为标题
```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
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
> 对于特别大的DataFrame，使用head方法只会选取前五行数据
如果指定了列序列，则DataFrame的列就会按照指定顺序进行排列：
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
如果指定的列在数据中找不到，就会在结果中产生缺失值：
```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'])
frame2
'''
	year	state	pop	debt
one	2000	Ohio	1.5	NaN
two	2001	Ohio	1.7	NaN
three	2002	Ohio	3.6	NaN
four	2001	Nevada	2.4	NaN
five	2002	Nevada	2.9	NaN
six	2003	Nevada	3.2	NaN
'''
```
获取某一列的值，可以使用类似字典标记的方式或属性的方式：
```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame2 = pd.DataFrame(data)
frame2.columns   # 获取所有的列   Index(['state', 'year', 'pop'], dtype='object')
frame2['state']  # 获取这一列中的值
frame2.state     # 使用属性的是方式
```
> `frame2[column]`适用于任何列的名，但是frame2.column只有在列名是一个合理的Python变量名时才适用。
获取某行数据也可以通过位置或名称的方式进行获取，比如用loc属性
```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame2 = pd.DataFrame(data)
frame2.loc[2]   # 行的索引
```
添加一个新的列与删除一个列
```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame2['eastern'] = frame2.state == 'Ohio'    # 添加一个bool值得新列
del frame2['eastern']    #  删除这个bool值的新列
```
另一种常见的数据形式是嵌套字典：
pandas就会被解释为：外层字典的键作为列，内层键则作为行索引：
```python
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
> 与字典：列表的形似不一样，嵌套字典类似一一对应的那种形式
你也可以使用类似NumPy数组的方法，对DataFrame进行行列互换：
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
根据指定的索引进行显示数据
```python
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = pd.DataFrame(pop,index=[2000,2001,2002])
frame3
```
## 索引对象
pandas的索引对象负责管理轴标签和其他元数据（比如轴名称等）。构建Series或DataFrame时，所用到的任何数组或其他序列的标签都会被转换成一个Index：
```python
 obj = pd.Series(range(3), index=['a', 'b', 'c'])
 index = obj.index
 index
 '''
 Index(['a', 'b', 'c'], dtype='object')
 '''
 我们拿到这索引对象是不能进行修改的，它是不可变对象
 Index的功能也类似一个固定大小的集合：
 在DataFrame中形成的二维表，如frame3.columns操作就可以拿到它的索引对象：Index(['Nevada', 'Ohio'], dtype='object')
 ## 基本功能
 pandas对象的一个重要方法是reindex,用该Series的reindex将会根据新索引进行重排。如果某个索引值当前不存在，就引入缺失值:
 ```python
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
'''
a   -5.3
b    7.2
c    3.6
d    4.5
e    NaN
dtype: float64
'''
```
对于时间序列这样的有序数据，重新索引时可能需要做一些插值处理。method选项即可达到此目的，例如，使用ffill可以实现前向值填充：
```python
In [95]: obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])

In [96]: obj3
Out[96]: 
0      blue
2    purple
4    yellow
dtype: object

In [97]: obj3.reindex(range(6), method='ffill')
Out[97]:   # 前向值的填充
0      blue
1      blue
2    purple
3    purple
4    yellow
5    yellow
dtype: object
```
借助DataFrame，reindex可以修改**行索引**：
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
**列**可以用columns关键字重新索引： 
```python
In [102]: states = ['Texas', 'Utah', 'California']

In [103]: frame.reindex(columns=states)
Out[103]: 
   Texas  Utah  California
a      1   NaN           2
c      4   NaN           5
d      7   NaN           8
```
## 丢弃指定轴上的项
```python
obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop(['d', 'c'])
'''
a    0.0
b    1.0
e    4.0
'''
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data.drop(['Ohio','Colorado'])     # 删除行
···
	one	two	three	four
Utah	8	9	10	11
New York	12	13	14	15
···
data.drop(columns=['one','two'])   # 删除列
'''
	  three	 four
Ohio	2 3
Colorado	6	7
Utah	10	11
New York	14	15
'''
```
> 以上的删除不会对原来的数据产生影响，你输出原来的数据还保持不变，但是当你在删除的时候，加上参数inplace=True就会真正的删除
## 索引、选取和过滤
Series索引（obj[...]）的工作方式类似于NumPy数组的索引，只不过Series的索引值不只是整数
```
obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj['b']  # 1.0
obj[['b', 'a', 'd']]
'''
b    1.0
a    0.0
d    3.0
'''
```
用切片可以对Series的相应部分进行设置：
```
obj['b':'c'] = 5
'''
a    0.0
b    5.0
c    5.0
d    3.0
'''
```
用一个值或序列对DataFrame进行索引其实就是获取一个或多个列：
```
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data[['three', 'one']]
'''
          three  one
Ohio          2    0
Colorado      6    4
Utah         10    8
New York     14   12
'''
```
通过切片或布尔型数组选取数据：
```python
data[:2]
'''
          one  two  three  four
Ohio        0    1      2     3
Colorado    4    5      6     7
'''
data[data['three'] > 5]  # 选择满足条件的行
'''
          one  two  three  four
Colorado    4    5      6     7
Utah        8    9     10    11
New York   12   13     14    15
'''
```
## 用loc和iloc进行选取
对于DataFrame的行的标签索引，我引入了特殊的标签运算符loc和iloc
```python
data.loc['Colorado', ['two', 'three']]   # 获取一行多列的数据
data.iloc[[1, 2], [3, 0, 1]]  # 获取索引是1，2行和3，0，1列的数据
'''
          one  two  three  four
Ohio        0    1      2     3
Colorado    4    5      6     7
Utah        8    9     10    11
New York   12   13     14    15
'''
data.loc[:'Utah', 'two']
'''
Ohio        0
Colorado    5
Utah        9
'''
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data.iloc[:, :3][data.three > 5]
'''
	one	two	three
Colorado	4	5	6
Utah	8	9	10
New York	12	13	14
'''
```
## 算术运算和数据对齐
1、对于Series相加的话，会将两者的索引取并集，总共有这么多的索引，而值就是对应的算术运算了
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
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                index=['Ohio', 'Texas', 'Colorado'])
'''
df1的输出值
            b    c    d
Ohio      0.0  1.0  2.0
Texas     3.0  4.0  5.0
Colorado  6.0  7.0  8.0
'''
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                 index=['Utah', 'Ohio', 'Texas', 'Oregon'])
'''
df2的输出值
         b     d     e
Utah    0.0   1.0   2.0
Ohio    3.0   4.0   5.0
Texas   6.0   7.0   8.0
Oregon  9.0  10.0  11.0
'''
df1 + df2   # 将它们相加时，没有重叠的位置就会产生NaN值
'''
            b   c     d   e
Colorado  NaN NaN   NaN NaN
Ohio      3.0 NaN   6.0 NaN
Oregon    NaN NaN   NaN NaN
Texas     9.0 NaN  12.0 NaN
Utah      NaN NaN   NaN NaN
'''
```
> 相加产生新的二维表的顺序列和行标题是按照排序规则展示的
## 在算术方法中填充值
使用df1的add方法，传入df2以及一个fill_value参数：
fill_value参数会将df中的值为NaN的时候，设置为这个填充的值，然后进行算术操作
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
计算一个二维数组与其某行之间的差：
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

默认情况下，DataFrame和Series之间的算术运算会将Series的索引匹配DataFrame的列，然后**沿着行一直向下广播**:【默认沿着行广播】
```python
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                    columns=list('bde'),
                    index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]
print(frame)
'''
          b     d     e
Utah    0.0   1.0   2.0
Ohio    3.0   4.0   5.0
Texas   6.0   7.0   8.0
Oregon  9.0  10.0  11.0
'''
print(series)
'''
b    0.0
d    1.0
e    2.0
'''
frame - series 
'''
	b	d	e
Utah	0.0	0.0	0.0
Ohio	3.0	3.0	3.0
Texas	6.0	6.0	6.0
Oregon	9.0	9.0	9.0
'''
```
如果某个索引值在DataFrame的列或Series的索引中找不到，则参与运算的两个对象就会被重新索引以形成并集：
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
如果你希望匹配行且在**列上广播**，则必须使用算术运算方法。例如：
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
NumPy的ufuncs（元素级数组方法）也可用于操作pandas对象：
```python
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),index=['Utah', 'Ohio', 'Texas', 'Oregon'])
np.abs(frame)
```
另一个常见的操作是，将函数应用到由各列所形成的一维数组上。DataFrame的apply方法即可实现此功能：
```python
f = lambda x: x.max() - x.min()
frame.apply(f)
'''
b    1.802165
d    1.684034
e    2.689627
'''
```
如果传递axis='columns'到apply，这个函数会在每行执行：
```python
frame.apply(f, axis='columns')
'''
Utah      0.998382
Ohio      2.521511
Texas     0.676115
Oregon    2.542656
'''
```
许多最为常见的数组统计功能都被实现成DataFrame的方法（如sum和mean），因此无需使用apply方法。
## 排序和排名
要对行或列索引进行排序（按字典顺序），可使用sort_index方法，它将返回一个已排序的新对象：
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
对于DataFrame，则可以根据任意一个轴上的索引进行排序：
```python
frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                 index=['three', 'one'],
                 columns=['d', 'a', 'b', 'c'])
frame.sort_index()
'''
       d  a  b  c
one    4  5  6  7
three  0  1  2  3
'''
frame.sort_index(axis=1)
'''
       a  b  c  d
three  1  2  3  0
one    5  6  7  4
'''
```
数据默认是按升序排序的，但也可以降序排序:frame.sort_index(axis=1, ascending=False)
若要按值对Series进行排序，可使用其sort_values方法：
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
在排序时，任何缺失值默认都会被放到Series的末尾：
```python
obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()
'''
4   -3.0
5    2.0
0    4.0
2    7.0
1    NaN
3    NaN
'''
```
当排序一个DataFrame时，你可能希望根据一个或多个列中的值进行排序:
```python
frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame.sort_values(by='b')
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
