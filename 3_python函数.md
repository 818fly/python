## 函数
如果你需要多次重复相同的操作或类似的代码，就非常值得写一个可复用的函数。
函数的声明使用关键字def，返回时使用关键字return:
```python
def my_function(x, y, z=1.5):
    if z >1:
        return z*(x+y)
    else:
        return z/(x+y)
```
**如果Python达到函数的尾部时仍然没有遇到return语句，就会自动返回None。**
每个函数都可以有位置参数和关键字参数。关键字参数最常用于指定默认值或可选参数。在上面的这个函数的例子中，x和y是位置参数，z是关键字参数。这意味着函数可以通过一下任意一种方式进行调用：
my_function(1,2,z=0.7)  #z的默认值会改变
my_function(1,2,3)  #z的默认值会失效
my_function(1,2)    #z会使用默认值进行安排
注意：如果我们写的函数参数中有关键字参数，那么关键字参数必须放在位置参数的后面（如果有的话）
## 命名空间、作用域和本地函数
在Python中描述变量作用域的名称是命名空间。在函数内部，任意变量都是默认分配到本地命名空间的。本地命名空间是在函数被调用时生成的，并立即由函数的参数填充。当函数执行结束后，本地命
名空间就会被销毁。考虑以下函数:
```python
def func():
    a =[]
    for i in range(5):
        a. append(i)
```
当调用func()时，空的列表a会被创建，五个元素被添加到列表，之后a会在函数退出时被销毁。
## 返回多个值
```python
def func():
   a = 1
   b = 2
   c = 3
   return a, b, c
a, b, c = func()
# 接受这个返回值还可以使用  values = func()
# values实际上就是包含了3个元素的元组
```
这里实际上返回的是对象，也就是元组，而元组之后又被拆包为多个结果变量。
## 函数是对象
由于Python的函数是对象，很多在其他语言中比较难的构造在Python中非常容易实现。假设我们正在做数据清洗，需要将一些变形应用到下列字符串列表中:
```python
states = [' Alabama', 'Georgia!', 'Georgia', 'georgia', 'FlorIda', 'south carolina##',  'West virginia?' ]
```
在清洗这种数据需要做的事情很多：去除空格，移除标点符号，调整适当的大小写。
方式一：使用内建的字符串方法，结合标准的正则表达式模块re
```python
import re
states = [' Alabama', 'Georgia!', 'Georgia', 'georgia', 'FlorIda', 'south carolina##',  'West virginia?' ]
def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[#?!]','',value)
        value = value.title()
        result.append(value)
    return result
clean_strings(states)
```
另一种会让你觉得有用的实现就是将特定的列表操作应用到某个字符串的集合上:
```python
import re
states = [' Alabama', 'Georgia!', 'Georgia', 'georgia', 'FlorIda', 'south carolina##',  'West virginia?' ]
def remove_punctuation(value):
    return re.sub('[!#?]','',value )
clean_ops = [str.strip, remove_punctuation, str.title]  # 这里面写的是函数的方法，在这里不需要加上括号的，在遍历调用的时候要加上括号，str是python的标量类型，含有一个处理字符串的方法

def clean_strings(strings, ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)  # 遍历的时候加上括号，表示分别调用列表中的函数
        result.append(value)
    return result
clean_strings(states, clean_ops)
```
像这种更为函数化的模式可以使你在更高层次上方便地修改字符串变换方法。clean_strings函数现在也具有更强的复用性和通用性。
## 匿名函数 lambda
匿名函数使用lambda关键字定义，该关键字仅表达“我们声明一个匿名函数“的意思，匿名函数注重的不是方法名，函数名，而是方法中的参数和方法体，所以在匿名函数中只需要写这两方面东西即可
```python
equiv_anon = lambda x: x * 2
equiv_anon(2)  # 调用lambda函数
```
匿名函数在数据分析中是非常方便的
```python
def apply_to_list(ints,f):
    return [f(x) for x in ints]
ints = [4, 0, 1, 5, 6]
apply_to_list(ints, lambda x: x*2)  # 将lambda函数传递给f函数
```
你也可以写成`[x*2 for x in ints]`，但是在这里我们能够简单地将一个自定义操作符传递给apply_to_list函数。
另一个例子，假设你想要根据字符串中不同字母的数量对一个字符串集合进行排序:
```python
listname = ['asd','aswa','a']
listname.sort(key=lambda x: len(x), reverse= True)
listname
```
## 生成器
通过一致的方式遍历序列，例如**列表中的对象或者文件中的一行行内容**，这是Python的一个重要特性。这个特性是通过**迭代器协议**来实现的，迭代器协议是一种**令对象可遍历的通用方式**。例如，遍历一个字典，获得字典的键:
```python
some_dict = {'a':1, 'b':2, 'c':3}
for key in some_dict:
    print(key)
```
当你写下for key in some_dict的语句时，Python解释器会自动尝试根据some_dict生成一个迭代器:iter(some_dict)。迭代器就是一种用于在上下文中(比如for循环)向Python解释器生成对象的对象。大部分以列表或列表型对象为参数的方法都可以**接收任意的迭代器对象**。包括内建方法比如min. max和sum,以及类型构造函数比如list和tuple:list(iter(some_dict)) 就会得到`['a','b','c']`

生成器是**构造新的可遍历对象的方式**。普通函数执行并一次返回单个结果，而生成器则返回一个多结果序列，在每一个元素产生之后暂停，直到下一个请求。
### 创建生成器
如需创建一个生成器，只需要在函数中将返回关键字return替换为yield关键字:
```python
def squares(n):
    for i in range(1,n):
        yield i**2
```
当你实际调用生成器时，代码并不会立即执行:例如调用这个生成器 squares(10)  会输出：<generator object squares at 0x000001EA84082E40>
直到你请求生成器里面的元素时，它才会执行上面定义的函数:
```python
gen = squares(10)
for i in gen:
    print(i)
```
## 生成器表达式
用生成器表达式来创建生成器是更加简单的方式。生成器表达式与列表、字典、集合的推导式很类似，创建一个生成器表达式，只需将列表推导式的中括号替换为小括号即可:
```python
gen =(x**2 for x in range(100))
gen
# out <generator object <genexpr> at 0x000001EA840BF4A0>    表示一个生成器产生了
```












