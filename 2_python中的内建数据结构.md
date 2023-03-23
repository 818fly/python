## 元组
### 元组的创建方式
元组是一种固定长度、不可变的Python对象序列。创建元组最简单的办法就是用逗号分隔序列值。
```python
tup = 1, 3, 4
```
当你通过更复杂的表达式来定义元组时，通常需要用括号将值包起来，例如下面这个例子。生成了元素是元组的元组:
```python
nested_ tup = (4, 5, 6)，(7, 8, 9)
#out: ((4, 5, 6)，(7, 8, 9))
```
你可以使用tuple函数将任意序列或迭代器转换为元组:
```python
tuple([4，0, 2])
tup = tup1e(' string') 
```
元组的元素可以通过中括号[]来获取，在大多数序列类型中都可以使用这个方法。
### 使用+号连接元组来生成更长的元组
注意这个不是对原有的元组进行修改，而是产生了一个新的元组
```python
(4, None,'foo') + (6，0) + ('bar',)
# out (4,None ,foo'6，0,bar' )
```
## 元组拆包
如果你想要将**元组型的表达式赋值给变量**，Python 会对等号右边的值进行拆包:
```python
tup = (4, 5, 6)
a, b, c = tup   # 我们使用逗号进行分隔这就是一个元组，进而可以对元组进行拆包直接赋值
```
使用这个功能可以轻松的对变量进行交换，在其它的语言中可以在交换变量的时候，需要使用中间变量temp
```python
a, b = 1, 2
b, a = a, b
```
拆包的一个常用场景就是遍历元组或列表组成的序列:
```python
seq = [(1, 2, 3)，(4, 5, 6)，(7, 8，9)]
for a, b，c in seq:
```
Python语言新增了一些更为高级的元组拆包功能，用于帮助你从元组的起始位置“采集“一些元素，舍弃一些你不需要的元素。这个功能使用特殊的语法`*rest`,用于在函数调用时获取任意长度的位置参数列表:
```python
values = 1, 2, 3, 4, 5
a, b, *rest = values
# out a = 1 b = 2 rest = [3, 4, 5]
```
注意：在大多数的工程师中，对于不需要的内容常用`*_`的写法，而不是`*rest`
## 元组的方法
由于元组是不可改变的序列，所以关于元组的实例方法很少，一个常用的方法就是count，(列表中也可用)，用于计量某个数值在元组中出现的次数:
```python
a = (1, 2, 3, 1, 1)
a.count(1)
```
## 列表
与元组不同，列表的长度是可变的，它所包含的内容也是可以修改的。你可以使用中括号[]或者**list类型函数**来定义列表:
```python
a_list = [1, 2, 3, 4]
tup = ('foo')
list函数在数据处理中常用于将**迭代器或者生成器**转化为列表:
```python
gen = range(10)
list(gen)
```
## 列表的操作
1. 使用append可以在列表的末尾增加元素
2. 使用insert方法可以将元素插入列表的指定位置 a_list.insert(index, info)
3. 使用pop将特定位置的元素移除并返回 ele = a_list.pop(2)
4. 使用remove方法移除，该方法会定位第一个符合要求的值并移除它
5. 使用关键字in可以检查一个值是否在列表中，not 关键字可以作用于in 的反义词
注意：与字典、集合(后面会介绍)相比，检查列表中是否包含一个值是非常缓慢的。这是因为Python在列表中进行了线性逐个扫描，而在字典和集合中Python是同时检查所有元素的(基于哈希表)。
## 连接和联合列表
与元组相似，两个列表可以使用+号连接
```python
[4, None, 'foo'] + [7, 8, (2, 3)]
# out [4, None, 'foo', 7, 8, (2, 3)]
 ```
如果你有一个已经定义的列表，你可以用extend方法向该列表添加多个元素: 
```python
x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])
# out [4, None, 'foo', 7, 8, (2, 3)]
 ```
注意：使用extend将元素添加到已经存在的列表比通过+连接列表好，尤其是在你需要构建-一个大型列表时，因为在使用+号的时候会创建新的列表，并且还要复制新对象
## 列表的排序
### sort
你可以调用列表的sort方法对列表进行内部排序(无须新建一个对象):
```python
a = [7, 2, 5, 1, 3]
a.sort()
a
```
sort有一些选项偶尔会派上用场。其中一项是传递一个二级排序key，用于生成排序值的函数。例如，我们可以通过字符串的长度进行排序:
```python
a = ['abc','asdcf','a']
a.sort(key=len)
a
```
### sorted
sorted方法可以针对通用序列产生一个排序后的拷贝，产生的是拷贝，不会影响原来的值
```python
a = ['abc','asdcf','a']
sorted(a)   # 这样回车会直接将结果进行返回，不需要写a，当你写a的时候还会显示a的原值
```
## 二分搜索和已排序列表的维护
注意：二分搜索的前提是列表是已排序好的，所以需要保持列表保持有序，对未排序的列表使用bisect的函数虽然不会报错，但可能会导致不正确的结果

内建的bisect模块实现了二分搜索：bisect. bisect会找到元素应当被插入的位置

已排序列表的插值：isect.insort将元素插入到相应位置
## 列表的切片
可以对大部分序列类型选取其子集，它的基本形式是将start：stop传入到索引符号[]中
```python
seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[2:5]
# out [3, 7, 5]
```
可以选取一个切片，对这个切片进行赋值
```python
seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[2:5] = [1, 1]
```
由于起始位置start的索引是包含的，而结束位置stop的索引并不包含，因此切片到的元素数量为stop-start。
### 切片的序号
start和stop是可以省略的，如果省略的话会默认传入序列的起始位置或结束位置
```python
seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[1:]  # stop省略 表示直接取到尾
seq[:3]  # start省略，表示直接从头开始
seq[:]  # start和stop都省略，表示取整个列表
```
负数的start和stop可以从列表的**尾部**进行索引
```python
seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[-6:-1]
# out [3, 7, 5, 6, 0]
```
**步进值step**可以在第二个冒号后面使用，意思是每隔多少个数取一个值:
```python
seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[::-1]  # 会直接将列表进行逆序排序
seq[::2]   # 会对整个列表每隔2个取一个值
```
## 内建序列函数
### enumerate
我们经常需要在遍历一个序列的同时追踪当前元素的索引。返回(i ,value)元组序列，其中value是元素的值，i是元素的索引

使用enumerate构造一个字典，将序列值映射到索引位置上。
```python
seq = [7, 2, 3, 7, 5, 6, 0, 1]
a_map = {}
for i, value in enumerate(seq):
    a_map[i] = value
a_map
```
### sorted 
返回一个新的已排序的列表
```python
seq = [7, 2, 3, 7, 5, 6, 0, 1]
sorted(seq)
```
### zip
zip将列表、元组或其他序列的元素配对，新建一个元组构成的列表:
```python
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zipped = zip(seq1,seq2)
list(zipped)
# out [('foo', 'one'), ('bar', 'two'), ('baz', 'three')]
```
zip可以处理任意长度的序列，它生成列表长度由最短的序列决定:
```python
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two']
zipped = zip(seq1,seq2)
list(zipped)
# out [('foo', 'one'), ('bar', 'two')]
```
zip的常用场景为同时遍历多个序列，有时候会和enumerate同时使用: 
```python
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zipped = zip(seq1,seq2)
for i,(value1,value2) in enumerate(zipped):
    print(str(i)+"-----"+value1,value2)
```
### reversed
reversed函数将序列的元素倒序排列
reversed(seq)
seq -- 要转换的序列，可以是 tuple, string, list 或 range。
## 字典
dict(字典)可能是Python内建数据结构中最重要的。它更为常用的名字是哈希表或者是关联数组。字典是拥有灵活尺寸的键值对集合，其中键和值都是Python对象。
用大括号{}是创建字典的一种方式，在字典中用逗号将键值对分隔:
```python
empty_dict = {}
d1 = {'a': 'some value', 'b':[1, 2, 3, 4]}
d1
```
访问、插入或设置字典中的元素，就像访问列表和元组中的元素一一样:
```python
d1 = {'a': 'some value', 'b':[1, 2, 3, 4]}
d1['a']
d1['c'] = 'insert value'
d1['a'] = 'xiugai'
d1
```
你可以用检查列表或元组中是否含有一个元素的相同语法来**检查字典是否含有一个键**:
```python
d1 = {'a': 'some value', 'b':[1, 2, 3, 4]}
'a' in d1   # 返回True
```
你可以使用del关键字或pop方法删除值，pop方法会在删除的同时返回被删的值，并删除键:
```python
d1 = {'a': 'some value', 'b':[1, 2, 3, 4]}
# del d1['a']  # 直接删除
d1.pop('b')   # 删除并返回被删除的值
```
keys和vaules方法分别会为你提供字典的键、值**迭代器**
你可以使用update方法将两个字典合并：
```python
d1 = {'a': 'some value', 'b':[1, 2, 3, 4]}
d1.update({'a':'update'})
d1
```
d1.update({'c': 12})   注意如果传给update的字典有与原字典相同的键，会将原字典的值进行覆盖
### 从序列中生成字典
通常情况下，你有两个序列，你想按照字典特性进行元素配对:
起初你会这么写
```python
empty_dict = {}
key_list = [1, 2]
value_list = ['a', 'b']
for key, value in zip(key_list,value_list):
    empty_dict[key] = value
empty_dict
```
再者，由于字典的本质是一个2-元组的集合。字典可以接受一个2元组的列表作为参数的：
```python
mapping = dict(zip(range(4),range(4)))  # 通过zip产生2元组列表
mapping
```
### 字典默认值
一个常见的场景是字典中的值集合通过设置，成为另一种集合，比如列表。字典的setdefault方法就是为了这个目的而产生的。
传统做法
```python
words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]   # 键是首字母，值是一个列表组成的词汇
    else:
        by_letter[letter].append(word) # 如果首字母存在，则利用by_letter[letter]拿到列表，在列表中追加值
by_letter
```
优秀做法
```python
words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}
for word in words:
    letter = word[0]
    by_letter.setdefault(letter,[]).append(word)
by_letter
```
说明：关于setdefault函数参数，第一个为key，第二个为默认值
### 有效的字典键类型
尽管字典的值可以是python的任意对象，但键必须是不可变对象，比如标量类型（整数、字符串、浮点数）或元组（且元组内的元素也是不可变对象）。通过判断一个对象是否可hash化，可以判断这个对象是否可以作为字典的键
```python
hash('a')  # 会返回hash值
hash([a])  #可变对象，会报错
```
## 集合
集合与数学中的集合概念是一样的，含有元素不能重复、无序的特性。**你可以认为集合也像字典，但是它只有键没有值**。集合含有两种创建方式，set函数和{}的方式：
```python
set([1,2,1,2])
# 或
{1，2，1，2}
# out {1,2}
```
集合支持数学上的集合操作，例如联合、交集、差集、对称差集。
```python
t | s          # t 和 s的并集  
t |= s         # 将t的内容设置为 t和 s的并集 
t & s          # t 和 s的交集 
t &= s         #将t的内容设置为t和 s的交集 
t – s          # 求差集（项在t中，但不在s中）
t -= s         #同理
t ^ s          # 对称差集（项在t或s中，但不会同时出现在二者中）
t ^= s         #同理
```
基本操作
```python
t.add(x)              # 将元素x加入集合a
t.pop()               # 移除任意元素，如果集合是空则抛出keyError
t.remove(x)           # 从集合中移除某个元素
len(s)                # set 的长度 
set1 = set([1,2,1,2,3])
set2 = set([1,2])
set1.issubset(set2)  #是否包含于的关系 False
set1.issuperset(set2)  #是否包含的关系  True   包含比包含于大
a.isdisjoint(b)      # a、b没有交集返回True
```
## 列表集合字典推导式
它允许你**过滤**一个容器的元素，用一种简明的表达式转换传递给过滤器的元素，从而生成一个新的列表。
表达式为:
```python
# [expr for val in collection if condition ]
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
list1 = [i.upper() for i in strings]
list1
```
集合与字典的推导式是列表推导式的自然拓展，用相似的方式生成集合与字典。字典推导式如下所示:
```python
# 表达式：dict_ comp = {key-expr : value-expr for value in collection if condition}
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
dict_comp = {key: value for key, value in enumerate(strings)}
dict_comp
```
集合推导式看起来很像列表推导式，只是中括号变成了大括号:
```python
set_ comp = {expr for value in collection if condition}
```
### 嵌套列表推导式
假如我们有一个包含列表的列表，你需要把包含两个a字母的单词打印出来
传统做法
```python
strings = [['aa', 'as', 'bat'], ['caar', 'dove', 'python']]
for i in strings:
    for j in i:
        if j.count('a') == 2:
            print(j)
```
使用嵌套列表推导式
```python
strings = [['aa', 'as', 'bat'], ['caar', 'dove', 'python']]
words_list = [word for i in strings for word in i if word.count('a')==2]
words_list
```




