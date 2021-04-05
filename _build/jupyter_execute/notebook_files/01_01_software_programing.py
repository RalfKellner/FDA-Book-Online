#!/usr/bin/env python
# coding: utf-8

# # Which Python?
# 
# Using Python for data analysis can be done in various ways. Python is an interpreted general-purpose programing language and all you basically need to execute python is an interpreter which can translate between the machine and you speaking in Python. However, especially for beginners, I recommend to install a complete Python distribution which manages the installation and use of Python. More specifically, I recommend to download and install the Anaconda package manager ([follow this link!](https://www.anaconda.com/products/individual)) which will not only install Python, but many software and language tools that can be helpful for analyzing data. Depending on your operation system, choose the appropriate installer and follow its guidance. 
# 
# Once your installation is finished, you can open the Anaconda Navigator as explained [here](https://docs.anaconda.com/anaconda/user-guide/getting-started/) and make sure that Juypter Notebook is installed, if not, you should do so. After Jupyter Notebook is installed, you may start it from the Anaconda Navigator menu or directly open a terminal and type "jupyter notebook". After that, you should see something like this cell:
# <br>
# 
# <img src="../images/jupyter_cell.png" alt="jupyter_cell" class="bg-primary" width="1000px">
# 
# Jupyter notebook is a web based application interface for documenting and among many other things can be used to execute and document python code. To execute python code simply click in the cell and write the following and click on the "Run" symbol above the cell.

# In[1]:


print('hello world!')


# ...and the above should happen. A recommendation is to work with shortcuts right from the start. Click somewhere except in the cell and press "h". An overview with keyboard shortcuts and corresponding actions should occur. For instance, we can press "b" if we are not inside a cell to generate a new cell below the current cell and if we are inside a cell when pressing "control + Enter" (at least on MacOS). Jupyter notebook does not only offer to execute code, we can also write plain text, math equations and many more if we convert the cell to markdown format. To learn more about markdown [you can read this](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html). Ok, now we can execute python code and learn how python code looks like.

# # Fundamentals for using Python
# 
# Phyton can be used in various ways and for various tasks. I will focus on the tasks which will be most important for us. The path I choose to give you an idea what you should learn in the beginning is the following.
# 
# <ol>
#     <li>Python as a calculator</li>
#     <li>Objects</li>
#     <li>Data structures</li>
#     <li>Functions</li>
#     <li>Control structures and loops</li>
#     <li>Object orientation</li>
#     <li>Modules</li>
# </ol>

# ## Python as a Calculator
# If you are not familiar with any programing languages so far, it might help as a start, that python is able to execute calculations as you know it from a simple calculator, for instance we can add $+$, subtract $-$, multiply $*$ divide $/$ or perform the exponential function with arbitrary base, for instance $2^3$. This is achieved by: 

# In[2]:


print(2 + 2)
print(2 - 4)
print(2 * 2)
print(2 / 4)
print(2**3)


# Besides typical math operators, Python already comes along with so called build-in functions. In general, a function is executed by writing the function name in addition with parentheses and, if necessary, putting arguments between parentheses, e.g.:

# In[3]:


abs(-2)


# Note python distinguishes between integer numbers and floating point numbers. So in some cases 2 is handled different to 2.0. Furthermore, as you already have seen in the very first example, Python is also able to handle data types different to numbers, i.e. strings....

# In[4]:


print('This is kind of cool...')


# However, if we want to combine numbers and strings we first need to convert numbers to strings because different data types can not be used together.

# In[5]:


print('This does not cause an error when printing the number: ' + str(2))
print('But this causes an error' + 2)


# Sometimes we may want to write something to our code which is not meant to be printed as output, but should serve as a documentation for us. This can be achieved by placing the #-Symbol before we write something and is called a comment. Comments are very important to properly code, especially when you are writing a long script including many different steps of calculation and analysis.

# In[26]:


#This is a comment and is not executed, for instance we can comment that 2 + 2 is calculated in the next line:
2 + 2


# ## Objects in Python
# 
# Python is an object-oriented programing language, so almost everything is an object in python. We will explain this in more detail at the end of this section. For the moment, you should know that we can assign numbers, strings or results of calculations to an object. This makes it much easier to continue calculations at a later stage.

# In[27]:


#assign the number -2.0 to the variable a
a = -2.0
print(a)

#assign the absolte value of a to the variable b
b = abs(a)
print(b)

#print the sum of a and b
print(a + b)


# ## Data Structures
# 
# Assigning single numbers to one particular object does not seem to be an appropriate approach for data analysis. Typically, we encounter a big amount of data points, e.g. stock prices of multiple stocks for a number of days. For a more efficient handling of data, certain data structures do exist. We start with the basic structures, namely:
# 
# <ol>
#     <li>Lists</li>
#     <li>Dictionaries</li>
#     <li>Tuples</li>
# </ol>

# **List:** A list is a collection of homogeneous or heterogeneous objects. Tpyically, lists are generated with square brackets or the *list*-function.

# In[3]:


first_list = [1, 2, 3]
second_list = list((1, 2, 3))

print(first_list)
print(second_list)


# Lists are very flexible and can be nested, this means a list can be a collection of list which are a collection of listst and so on, e.g.:

# In[4]:


one_list = [1, 2, 3]
another_list = ['hello_world']

overall_list = [one_list, another_list]

print(overall_list)


# To access elements of a list, we can use indexing, starting the count at $0$: 

# In[12]:


#create list
my_list = [1, 2, 3, 4, 5, 6]

#print first value in list
print(my_list[0])

#print third value in list
print(my_list[2])

#print the first three entries in list
print(my_list[:3])

#print the last three entries in list
print(my_list[-3:])

#print all values except the last entry
print(my_list[:-1])

#print the last entry in list
print(my_list[-1])


# As indicated earlier, almost everything is an object in python. Lists are objects and objects usually have certain methods that can be accessed by the objects name and a dot. Here, you can see some usefull examples of list-methods:

# In[23]:


my_list = [5, 3, 2, 7, 9]

#add a value to a list
my_list.append(7)
print(my_list)

#count how many 7s are in the list
print(my_list.count(7))

#get the index at which position we find the value of 3
print(my_list.index(3))

#sort the list
my_list.sort()
print(my_list)

#reverse the list
my_list.reverse()
print(my_list)


# Lists are mutable, this means single elements can be accessed and changed by us:

# In[25]:


print(my_list[3])
my_list[3] = 99
print(my_list[3])


# One very useful and pythonic thing is list comprehensions. By a list comprehension, we can do the same thing with every element in the list and generate a list of the result of these operations. Assume we want to add the number $2$ to each element in the list. This can be done by:

# In[54]:


#create a list
my_list = [1, 2, 3]
print(my_list)

#add 2 two every element in the list 
add_two = [x + 2 for x in my_list]
print(add_two)

#x serves as a placeholder for each element in the list which gets processed


# **Tuple:**
# Unlike lists, tuples are not mutable, but besides that look similar to lists. Tuples are created with round brackets or with the *tuple*-function. Indexing can be done a with lists.

# In[34]:


my_tuple = (1, 2, 3)

#print the second element of the tuple
print(my_tuple[1])

#print the first two elements of the tuple
print(my_tuple[:2])

#count the elements in a tuple
print(my_tuple.count(2))

#get the index for an element
print(my_tuple.index(2))

#what is not possible is to append further elements or to alter existing elements
my_tuple[1] = 99


# A useful attribute of tuples is unpacking and better explained by example.

# In[41]:


#create a tuple with two numbers
one_two_tuple = (1, 2)
print(one_two_tuple)

#now unpack the two values and assign these values to objects 'one' and 'two'
one, two = one_two_tuple
print(one)
print(two)


# **Dictionary:** Dictionaries are mapping type data types and very useful. A dictionary is a collection of key-value-pairs and is created by curly brackets or the *dict*-function. Dictionaries are mutable objects and elements of dictionaries are accessed by the key, not be indexing. Dictionaries come along with useful methods such as evaluating if a key exists in the dictionary, get a list of keys, values or key-value pairs.

# In[47]:


my_dict = {'a': 1, 'b': 2}
print(my_dict)

#access the element a
print(my_dict['a'])
#this can also be done, using the get-method
print(my_dict.get('a'))

#change the value of b
my_dict['b'] = 20
print(my_dict['b'])

#add c and its value
my_dict['c'] = 5
print(my_dict)

#analyze if the key 'c' is in the dictionary
print('c' in my_dict)

#see what happens if the key is not in the dictionary
print('d' in my_dict)

#get the list of keys in the dictionary
print(my_dict.keys())

#get the list of values in the dictionary
print(my_dict.values())

#get all key-value pairs
print(my_dict.items())


# More data types and methods exist, but knowing these three will be enough for the start. Note that functions exist which can be executed for all data types. For instance, if you are not sure what data type you observe, you can use the *type*-function. Or, use the *len*-function for finding the number of elements in an object, or use *del* if you want to delete an object. The more you work with python, the more useful functions you will get to learn. This is not something to worry about in the beginning because if naturally evolves during work.

# In[58]:


a_list = [4, 2, 5]
a_tuple = (4, 2, 5)
a_dict = {'a': 4, 'b': 2, 'c': 5}

#what types are these objects
print(type(a_list))
print(type(a_tuple))
print(type(a_dict))

#how many elements are stored in these objects
print(len(a_list))
print(len(a_tuple))
print(len(a_dict))

#let us delete these objects
del(a_list)
del(a_tuple)
del(a_dict)

print(a_list)


# ## Functions
# When we want to execute the same task multiple times, we can define a function to do the job and save us some time and code. Functions are a sequence of instructions performing a task bundled in a unit. You already saw some built-in function like *type*, *len* or *del* which perform the tasks of showing the object type, the number of elements in an object or to delete the object. Luckily, python also gives us the opportunity to define our own functions, if we want to perform some tasks which are not captured by existing functions. To define a function, we use the keyword 'def' which is following by a function-name that we choose and round brackets. After the brackets a colon follows with a linebreak. The so called body is defined in an indented list of instructions. Functions can handle input and are able to output results (both is optional). Let us define the *add_two*-function which adds a value of $2$ to an input variable x and returns the result.

# In[59]:


def add_two(x):
    return x + 2

#execute the function
print(add_two(2))
print(add_two(5))


# As you can see, we use the return keyword for returning results of the function body. x serves as a placeholder for which we also could have used any other arbitrary name. We can also set a default value for x in the squared brackets which is used as long as we do not provide additional input. We can also return more than one number or print something to the screen during execution.

# In[7]:


def add_two_new(a_number = 42):
    print('We add 2 to the number: ' + str(a_number) + '!')
    return a_number, a_number + 2

#execute with default value
print(add_two_new())

#execute with 21
print(add_two_new(21))


# You may have noticed that the function returns a tuple. This can be used when we want to save the results as objects again.

# In[67]:


#either save the tuple as a whole
input_output = add_two_new(a_number = 21)
print(input_output)

#or use unpacking to save the result separately
inpt, outpt = add_two_new(a_number = 3)
print(inpt)
print(outpt)


# It is also important to know that objects which are assigned during function execution are local. This means they are available during execution, but not globally after exection. For instance:

# In[74]:


def local_var():
    #assign a value of 1 to the object local
    local = 1
    #return this value
    print(local)
    
#if we execute the function, pyhton knows during execution local has a value of 1
local_var()

#but after execution the value is no longer known
print(local)


# This relates what is called scope in python. We have something called a local and a global scope (and two other scopes which are not important to understand the point made here). The local scope is where python first looks for  when it encounters the function during execution. For instance in our example, the function needs to find the object local to print its value when we execute the function *local_var*, it first starts its search in the local scope of the function. In our example python finds the object local and prints its value found in the local scope. Given, the object local is not found in the local scope, python searches (in the enclosing scope and then) in the global scope afterwards. If it finds the local object here, it prints its value. Let us alter the example to see this:

# In[75]:


local = 1

def local_var():
    print(local)
    
local_var()


# If the object local is defined globally and local, python will use the local scope within the function and the global scope outside the function.

# In[76]:


local = 2

def local_var():
    local = 1
    print(local)
    
local_var()
print(local)


# ...pretty theoretical, but very useful to avoid painful errors in the future. Good standard is to document functions using docstrings. These typically directly follow after declaring the function name and are written within three quotation marks.

# In[77]:


def local_var():
    """This function assigns a value of 1 to an object called local
    and prints this value!"""
    
    local = 1
    print(local)


# Using the *help*-function, you can access the docstrings of functions. This is very helpful, when we work with functions written by other users.

# In[80]:


help(local_var)


# ## Control Structures and Looping
# 
# **Control structures**: In many situations, our operations depend on a condition, e.g., "...if the weather is good, I will go out, if not, I will stay at home...". In Python, we work with *if*, *if-else* or *if-elif-else* statements to include state dependent operations. Note that as for functions, the part of the code which belongs to the conditional statement is marked by indentation. Again let us learn by examples:

# In[16]:


#first, we just use an if statement
a = 1
if a == 1:
    print('a is one!')

#but if a is not 1, then nothing would happen so
#and combined with an else statement
if a == 2:
    print('a is two!')
else:
    print('the value of a is not two!')

#and if we have more than two conditions, we can handle
#this with the elif statement
today = 'friday'

if today in ['monday', 'tuesday', 'wednesday', 'thursday']:
    print('oh no, still work to do...')
elif today == 'friday':
    print('thank god its friday!')
else:
    print('it seems weekend has arrived...')


# When we evaluate conditions, we use comparison which return booleans (True or False). See the following exeamples of various ways for making such comparisons:

# In[19]:


a = 10
b = 15

#evaluate if a is equal to b
print(a == b)

#evaluate if a is not equal to b
print(a != b)

#is a smaller than b
print(a < b)

#is a smaller or equal to b
print(a <= b)

#is a greater than b
print(a > b)

#is a greater than or equal to b
print(a >= b)

#multiple comparisons can be combined by and/or statements
print((a != b) and (a < b))
print((a == b) and (a < b))
print((a == b) or (a < b))

#Negation can also be analyzed by the not statement
print(not(a == b))

#Use in to analyze if a certain element is in an (iterable) object
my_list = [1, 2, 3]
print(1 in my_list)
print(4 in my_list)


# **Looping:** Sometimes we may want to execute operations for a sequence of elements. This can be done by loops, whereby, Python mainly offers the *for* and the *while* loop. The for loops iterates over an object and executes the body marked by indentation for every element in the object.

# In[60]:


persons = ['Peter', 'Andrea', 'John']

for i in [0, 1, 2]:
    print(persons[i])


# Very often, the *range* function is used in Python instead of manualy coding a list to iterate over. The range function starts by default at $0$ and counts stepwise to a stop value. Take a look at the different options (and please ignore why we put *list* around the range function here):

# In[54]:


print(list(range(5)))
print(list(range(0, 5, 1)))
print(list(range(10, -12, -2)))


# So our first example with the *range* function looks like:

# In[61]:


persons = ['Peter', 'Andrea', 'John']

for i in range(3):
    print(persons[i])


# However, Python offers an even more intuitive method, we can also conduct the first example with:

# In[62]:


persons = ['Peter', 'Andrea', 'John']

for person in persons:
    print(person)


# And if we want to gain the information of the position in the list, we can use *enumerate*:

# In[63]:


persons = ['Peter', 'Andrea', 'John']

for i, person in enumerate(persons):
    print(str(i) + ' ' + person)


# If we want to loop over multiple iterables, the *zip* function is our go-to-guy in Python:

# In[66]:


persons = ['Peter', 'Andrea', 'John']
ages = [23, 25, 24]

for person, age in zip(persons, ages):
    print(person + ' is ' + str(age) + ' years old.')


# This also works with more than two iterables:

# In[68]:


persons = ['Peter', 'Andrea', 'John']
ages = [23, 25, 24]
cities = ['Munich', 'Passau', 'Regensburg']

for person, age, city in zip(persons, ages, cities):
    print(person + ' is ' + str(age) + ' years old ' + 'and is from ' + city + '.')


# The *while* loop repeatedly executes its body as long as a condition is not fullfilled.

# In[69]:


a = 0
while a <= 5:
    print(a)
    a = a + 1


# Be careful that the stopping condition can be achieved, otherwise, the code block runs forever. Especially, when using *for* and *while* loops, the *break* and *continue* signal word can be of use. The former immediately terminates exectution, while the latter skips the current step:

# In[80]:


a = 0
while True:
    print(a)
    a += 1
    if a > 5:
        break


# In[81]:


for a in range(5):
    if a == 3:
        print('Skipping this value...')
        continue
    print(a)


# ## Object Orientation
# As stated earlier, almost everythin in Python is an object, but what does that mean? Objects typically include different types of information, more concrete attributes and methods. We also have seen that different types of objects exists, e.g. integer numbers, strings, floating numbers, so even an integer of $2$ is not the same as the integer $3$ it is the same type. This is why we can add or multiply these two numbers, but we can not add a string "hello" to the integer $2$. Behind these observations stands the concept of object orientation. Objects of the same type follow the same blueprint and objects of different types have different blueprints. The blueprint is called a class in programing. A class is the prototype for objects of the same type. It is not important for the task of data analysis to be an expert in object oriented programing, but it is useful for better understanding some functionalities when working with data analysis applications at a later stage. However, I try to be brief at this point and may omit some information. We directly take a look at an example of a class called *Student* that we define to gather some fundamental knowledge for classes and objects in Python. Regarding its syntax, a class definition looks similar to the definition of a function. The definition starts with *class* and the class name we choose. The content of the class is defined in the indented body. Typically, objects of a class have certain attributes and methods. In Python attributes can be assigned when an object is created on the basis of a class definition, so at initialization. This is handled by the initialization method, which is a special function, called a **magic method** in Pyhton (for the moment, you do not need to care what magic methods are). An important keyword here is self, which simply means we refer to the instance that is created by the class. Typically, self comes first when defining class methods. Ok, but now let us make this concrete and learn from the example.
# 
# We want to define the class "Student" which should contain information regarding the subject, the age and the registration number. These are attributes. Furthermore, we define a method which enables the student to say hello.

# In[27]:


class Student():
    
    #initialization method
    def __init__(self, subject, age, registration_nr):
        self.subject = subject
        self.age = age
        self.registration_nr = registration_nr
        
    #define a method to say hello to a certain person
    def say_hello(self, name):
        #now let us say hello and do not get confused by the 
        #format syntax, this is string formating and we also could
        #have written print('Hello' + name + ', how are you?')
        #String formating is just more elegant to me
        print('Hello {}, how are you?'.format(name))


# In[34]:


#Let us define Peter and say hello to Anna
Peter = Student("math", 24, "123456")
Peter.say_hello('Anna')
#Let us take a look at Peter's information
print(Peter.subject)
print(Peter.age)
print(Peter.registration_nr)

print('-' * 30)

#Let us define Anna
Anna = Student('economics', 27, "654321")
Anna.say_hello('Peter')
#Let us take a look at Anna's information
print(Anna.subject)
print(Anna.age)
print(Anna.registration_nr)


# I hope this an example how classes work and what is meant by an object, its attributes and methods. There is so much more to say about this topic, but for the moment, the most important knowledge is that objects can share commonalities defined by their class definition. An object's attributes and methods can be accessed the objects name, a dot and the attributes or methods name (Attributes without brackets, methods with brackets). Furthermore, we can look inside an object using the *dir* function:

# In[35]:


dir(Peter)


# So right now, you should not be surprised when we, for instance, define a string, look inside the object and find it comes along with different methods...

# In[38]:


hello = 'hello world'
dir(hello)


# ...to find out what a certain method does, you can use the *help* function, e.g.,

# In[37]:


help(hello.split)


# In[39]:


hello.split()


# This is enough to know for your start. Later, we will get to know different object types. With the presented procedure of the last cells, we have an idea, how we can find out, what information certain object contain and how to find out, what certain methods of the objects do.

# ## Modules and Packages
# Despite its intuitive syntax and elegant ways of handling things, Python offers another advantage which is probably most important for its popularity. The existence of user-friendly, high-quality packages for data analysis. Imagine, we want to use two functions, the say-hello- and the say-goodby-function, for all our analysis. Instead of defining these to functions for every analysis, we can define them in one script and import this script for each of our analyses. The script is called a **module** and is a collection of statements and definitions. For instance, write the following code in a text editor of your choice and save it with the ".py" ending (not the ".ipynb" ending of jupyter notebook files) in the directory of your default anaconda directory under the name "my_module.py".

# In[1]:


#define the say-hello-function
def say_hello():
    print('hello!')
    
#define the say-goodbye-function
def say_goodbye():
    print('goodbye!')


# Now, open a juypter notebook and type the following:

# In[3]:


import my_module

my_module.say_hello()
my_module.say_goodbye()


# I hope you get the idea what a module is. As indicated by the dot usage of functions, the module can be seen as an object and the functions defined in the my_module script are its methods. Two further options exist for importing content of my_module. First, if you want import all methods and directly use the function you can do this:

# In[6]:


from my_module import *

say_hello()
say_goodbye()


# Or, if you want to directly import certain methods from my_module, you can do this:

# In[7]:


from my_module import say_hello

say_hello()


# Modules can have submodules with a nested structure. A collection of modules with a certain structure is called a **package** and this is were it gets very interesting for Python users. A multitude of packages exists including an immense amount of efficiently written methods for various tasks. For us, very important packages are [numpy](https://numpy.org/) among others offering a broad range of calculus methods, [pandas](https://pandas.pydata.org/) for data handling, [scikit](https://scikit-learn.org/stable/) for machine learning methods, [tensorflow](https://www.tensorflow.org/) for deep learning, [plotly](https://plotly.com/) for interactive visualization or [matplotlib](https://matplotlib.org/) for static visualization. All packages exhibit a good documentation and are worth exploring, however, in my experience, the best ways of learning the usage of these packages is to do this with a hands-on-approach which we will follow during the course. If you installed Anaconda, a few of these packages are installed with the Python base package. However, you may encounter situations in which you need to install external packages. Packages are managed by online repositories, i.e. Conda or Pypi. If you want to install a package from one of these repositories, the best way is to open the terminal and type *conda* (or *pip*) and the package name. Further options can be used for installation, but this is not too important in the beginning. For more information see [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/installing-with-conda.html). To give you an idea, how we will work with packages, we close this chapter with an example which shows you how to import the package numpy and use it to define two matrices and multiply these two. 

# In[14]:


import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[4, 3], [2, 1]])

print(A)
print(B)

np.matmul(A, B)

