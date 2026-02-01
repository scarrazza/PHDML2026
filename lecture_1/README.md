# Introduction to Python for ML

Modern applications of machine learning require the knowledge of Python, a high
level language, simple to code and be interfaced with low level languages, such as C.

## Environment preparation

For this tutorials we have setup in the laboratory machines an anaconda
environment with all required packages.

In order to activate the environment open a terminal and do:

```bash
module load python3/anaconda
```

This will select a `python` 3.x interpreter from the [anaconda distribution](https://www.anaconda.com/).

Then, to enable the tutorial environment with all packages relevant for ML do:
```bash
source activate DeepLearning
```

If you are working in our own machine we suggest to install [anaconda3](https://www.anaconda.com/) and then perform the following installations with:
```bash
conda install tensorflow keras numpy matplotlib pandas scikit-learn seaborn ipython-notebook
```

## Python basics

In this tutorial we will use `python` through `jupyter` notebooks. In order to start a session open a terminal and do:
```bash
jupyter notebook
```
This command will open a browser with an interactive python3 session.

Commands are executed cell by cell by pressing SHIFT+ENTER.

### Data types

Python is a dynamic typing language containing variables, lists and dictionaries.

Some examples of variables:
```python
a = 5       # int
b = 6.5     # float
c = 'hello' # str
```

In order to print to screen the values of each variable use the function `print`, e.g.:
```print
print(a)
>> 5
```

Some examples of lists:
```python
# list example
mylist = [2, 3, 4, 5]
mylist2 = [] # empty list

# appending elements to list
mylist.append(6) # mylist = [2, 3, 4, 5, 6]
```
Accessing a list item:
```python
print(mylist[0])
>> 2
```

Examples of dictionary:
```python
# dict example
mydict = {'key_1': value1, 'key_2': value2}
mydict2 = {} # empty dict

# append new key-value
mydict['key_3'] = value3
```
Accessing a dict item:
```python
print(mydic['key_1'])
>> value1
```


#### Exercises:
1. Create two `float` variables assigned to random numbers perform basic math operations such as +, -, *, /.
2. Create a list with five integers.
3. Append 3 floats to an empty list.
4. Create a dictionary with the following keys `('name', 'loss', 'weights')`. Initialize the dictionary with the following default values `('neuralnet', 0.12, [10,25,5])`. Print to screen the value of `weights`. Assign to `loss` the value 10.

### Cycles

Similarly to other programming languages, Python provides `while` and `for` loops, e.g.:
```python
# example of for loop
s = 0
l = [1, 5, 10, 20, 50]
for i in range(len(l)):
    s += l[i]
```
and
```python
# example of while loop
i = 1
while i < 6:
    if i == 3:
        break
    elif i == 2:
        i += 1
        continue
    i += 1
```

#### Exercises:

1. Given the list [2,6,3,8,9,11,-5] compute the mean value using a `for` loop.
2. Define a list using the expression `2 ** n` with `n = 10`.

### Functions

Functions are defined by the `def` keyword and follow the indentation structure for command blocks.
```python
# example of function in python
def factorial(x):
    if x == 0:
        return 1
    else:
        return x * factorial(x-1)
```

#### Exercise:

Write a function which returns the average of list [2,6,3,8,9,11,-5].

### Classes

Classes in python are flexible objects for custom data structures.
Their representation is based on a dictionary.
```python
class Variable:
    def __init__(self, name):
        self.name = name

    def sample(self):
        raise NotImplementedError()
```

Inheritance is also easily achieved with:
```python
import random

class Normal(Variable):
    """Standard normal random variable"""
    def sample(self):
        return random.normalvariate(0, 1)
```

#### Exercise:

Implement the `Normal` class and create a list of samples with 100 elements.

## Numpy basics

Numpy has several methods for linear algebra manipulation, e.g.:
```python
# numpy arrays
import numpy as np
b = np.array([1,2,3])
print(a.shape)
>> (3,)

# matrix
a = np.array([[1,2,-3],[0,1,0],[0,0,1]])

# dot product
np.dot(a, b)
# or
a.dot(b)
```
Numpy provides access to several numerical implementations.

#### Exercise:

Implement the dot product in a pure python function, using `for` loops.
Benchmark the performance of numpy.dot and the custom `dot` function.

## Matplotlib basics

Simple plotting example.
```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.linspace(0.0, 5.0, 100)

plt.plot(t1, f(t1), 'bo')
plt.xlabel("x")
plt.ylabel("y")
plt.title('example')
```

#### Exercise:

Extend the previous code with a second curve `f2(x)=cos(2*pi*x)`.

## pandas basics

Pandas provides a easy and quick way to dataframe manipulation. e.g.:
```python
import pandas as pd

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)
```
another example
```python
dates = pd.date_range('20130101', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

print(df)
```

## scikit-learn basics

Scikit-learn has many interesting features and great documentation.

#### Exercise:

Apply ordinary least squares through linear regression model.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def true_fun(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)

n_samples = 30
degrees = [1, 4, 15]

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X.reshape(-1,1), y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X.reshape(-1,1), y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test.reshape(-1,1)), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
```

# Keras quick overview

## What is keras?

Is a high-level neural networks API, written in Python, simple to manipulate and to extend.

It proposed a simplified interface to three ML engine backends: TensorFlow, CNTK and Theano.

## Sequential models

Keras create a sequential model, i.e. models where the data input is propagated sequentially from the first layer to a next layer.

```python
# loading the sequential model module
from tensorflow.keras import models

model = models.Sequential() # load an empty sequential container
```

## Dense layer

Dense is a layer based on a weight matrix.

Lets create a dense layer with just 1 parameter:
```python
from tensorflow.keras import layers

model.add(layers.Dense(units=1, input_dim=1))
```

## Visualizing model

It is possible to visualize the model setup by calling the method `.summary()`:

```python
print(model.summary())
```

## Producing model predictions

The method `.predict` evaluates the model prediction for a given `x` input.
```python
model.predict(x=[1])
```

## Getting weights from model

Furthermore the model parameters/weights/variables are available by calling the method `.get_weights()`.
```python
w = model.get_weights()
print(w)
```

## Compile model with loss and optimizer

We will see in the next lesson that a model can be combined/compiled with a loss function and optimizer, so the model will be ready for training.
```python
model.compile(loss='mse',
              optimizer='sgd')
```
