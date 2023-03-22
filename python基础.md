## 一切皆为对象
Python语言的一一个重要特征就是对象模型的一致性。每一个数值、字符串、数据结构、函数、类、模块以及所有存在于Python解释器中的事物，都是Python对象。每个对象都会关联到一种类型(例如字符串、函数)和内部数据。在实践中，一切皆为对象使得
语言非常灵活，甚至函数也可以被当作对象来操作。
例如，你输入一个 a = 1 都有与之对应的方法，如得到此数的实部虚部
## 变量和参数传递
在python中对一个变量（或者变量名）赋值时，你就创建了一个指向等号右边对象的引用在你拥有一个列表的时候 a = [1, 2, 3]，你将它的值赋值给一个新的变量b，b=a。在某些语言中，会是数据[1, 2，3]被拷贝的过程。在Python中，a和b实际上是指向了相同的对象，即原来的[1，2，3]。两个引用指向同一个对象
一般性说明：赋值也称为绑定，这是因为我们将一个变量名绑定到了一个对象上。
## isinstance
了解对象的类型非常重，你可以使用isinstance函数来检查一个对象是否是特定类型的实例。
```java
a = 1
isinstance (a,int)
```
isinstance接受-个包含类型的元组，你可以检查对象的类型是否在元组中的类型中
## is关键字
检查两个引用是否指向同一个对象
```java
a = [1, 2, 3]
b = a
c = list(a)
print(a is b)  #True
print(a is not c)   #True
```
因为list函数总是创建一个新的python列表，我们可以确定c与a是不同的。is和==是不同的，因为在这种情况下我们可以得到 a==c 返回的是True
is和is not的常用之处是检查一个变量是否为None
## 可变对象与不可变对象
Python中的大部分对象，例如列表、字典、NumPy数组都是可变对象，大多数用户定义的类型(类)也是可变的。可变对象中包含的对象和值是可以被修改的:
```java
a_list = ['foo', 1, [4, 5]]
a_list[0] = 'food'
a_list
```
还有其他一些对象是不可变的，比如字符串、元组:
```java
str1 = 'abc'
print(str1[0:1]='d')
str1 = 'a'
print(str1)
```
第一个我们试图将str1的字符串的第一个字母进行修改，会报错不允许进行修改，但是要说明的是下面的str1='a'这个不是修改语句，这样的做法只是将str1变量指向的地址改变了
## python标量类型
Python的标准库中拥有一个小的内建类型集合，用来处理数值数据、字符串、布尔值以及日期和时间。这类的“单值”类型有时被称为标量类型
1. None 表示的是python的'null'值（是NoneType类型的唯一实例)
2. Str 表示字符串类型
3. bytes 表示原生ASCII字节（或者Unicode编码字节）
4. float 表示双精度64位浮点数值（注意python没有double类型）
5. bool 表示True或False
6. int 表示任意精度无符号整数
## 类型转换
str、bool、int和float既是数据类型，同时也是可以将其他数据转换为这些类型的函数
## None
None是Python的null 值类型。如果一个函数没有显式地返回值，则它会隐式地返回None:
None还可以作为一个常用的函数参数默认值:
```java
def add_and_maybe_multiply(a, b, c=None):
    if c is None:
        return a+b
    else:
        return a+b+c
```
## 日期与时间
Python中内建的datetime模块，提供了datetime、data和time类型。可能正如你想象的，datetime类型是包含日期和时间信息的，最常用的方法是:
```python
from datetime import datetime, date, time
dt = datetime(2011, 10，29，20，30，21)
# 当你有了一个时间对象之后，你就可以获取这个时间对象中day、minute、date和time对象
dt.day
dt.minute
dt.date()
dt.time()
```
**1. datetime——> str**
```python
dt.strftime('%m/%d/%Y %H:%M')
```
还有一种，可以将字符串类型的时间转了datetime对象
**2. str——>datetime**
```python
datetime.strptime('2009-10-31'，'%Y-%m-%d')
# 输出的内容是：datetime.datetime(2009, 10, 31, 0, 0)
```
注意：p字母表示的parse解析，解析的过程其实就是根据我们后面的匹配格式进行匹配的，然后抽取到时间点
当你在聚合或分组时间序列数据时，会常常用到替代datetime时间序列中的一-些值，比如将分钟、秒替换为0:
```python
dt . replace (minute=0，second=0)
datetime .datetime(2011, 10，29, 20，0)
```
由于datetime.datetime是不可变类型，以上的方法都是产生新的对象。
两个不同的datetime对象会产生一个datatime.timedelta类型的对象: 我们将两个时间类型进行相减，会得到这两个时间对象的时间差
同理，我们可以将一个datetime对象加上这个时间差得到另一个时间对象
## python中的且或表达
在条件判断的过程中，使用and和or表示且或的关系，同时，它与其他语言中是一样的也会出现短路的情况
**补充break，break会结束离它最近的那一层循环**
## pass
pass就是Python中的“什么都不做”的语句。它用于在代码段中表示不执行任何操作(或者是作为还没有实现的代码占位符) ;之所以需要它，是因为Python使用了缩进来分隔代码块
## range
python中的range函数返回的是一个迭代器，该迭代器返回的是一个等差整数序列，起始、结尾、步进(可以是负的)可以传参给range函数：range(0,20,2)range产生的整数包含起始但不包含结尾。
range常用于根据序列的索引遍历序列:
```python
seq = [1，2, 3，4]
for i in range(len(seq)):
    val = seq[i]
```
尽管你可以使用函数，比如list函数将range产生的整数存储在其他数据结构，但通常默认的迭代器形式就是你想要的。以下代码会将0到99,999中所有可以被3或5整除的整数相加:
```python
sum=0
for i in range ( 1000000):
# %是求模操作符
    ifi%3==Oori%5==0:
    sum += i
```
虽然range产生的序列可以是任意大小，但在任意给定时间内的内存使用量是非常小的。
## 三元运算符
在Java、JavaScript、C等其他其他代码程序中，三元运算符一般是使用“? :”组合，但是在python中不是这样的，它是使用“if else”组合：
```python
# 语法：语句1 if 条件表达式 else 语句2
return '输出成功' if res=='success' else '输出失败'
```
虽然我们可以使用三元表达式来压缩代码量，但请注意如果条件以及真假表达式非常复杂，可能会牺牲可读性。










