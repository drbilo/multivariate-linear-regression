# Multivariate Linear Regression w. Gradient Descent

## Introduction

Last time I talked about [Simple Linear Regression](https://github.com/drbilo/linear_regression) which is when we predict a y value given a single feature variable x. However data is rarely that simple and we often can have many variables we can use. For example, what if we want to predict the cost of a house and we have access to the size, number of bedrooms, number of bathrooms, age etc. For this kind of prediction we need to use Multivariate Linear Regression.

#### Updated Hypothesis Function

Luckily to do this it doesn't require too much to change compared to Simple Linear Regression. The updated hyptohesis function looks like so:

![alt text](https://www.dropbox.com/s/t733qinzqspjtgm/mvlr_hypothesisfunction.png?raw=1 "Hypothesis Function")

To calculate this efficiently, we can use matrix multiplication which is used in Linear Algebra. Using this method our hypothesis function looks like this:

![alt text](https://www.dropbox.com/s/t3gwa7358r130uw/mvlr_matrixhypothesis.png?raw=1 "Matrix hypothesis")

To do this succesfully, we have to match our two matricies in size so that we can perform matrix multiplication. We acheive this by setting our first feature value ![alt text](http://www.sciweavers.org/tex2img.php?eq=x_%7B0%7D%5E%7Bi%7D%20%3D%201&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0 "x0 feature setting"). To explain more simply, the amount of columns in our training matrix must match the amount of rows in our theta vector.

#### Cost Function

The cost function remains similar to the one used in simple linear regression althogh an updated one using matrix operations looks like this:

![alt text](https://www.dropbox.com/s/gxqeu7kemtcjvpk/costfunction.png?raw=1 "vectorized cost function")

This is implemented in my code as:

```python
def cost_function(X, y, theta):
    m = y.size
    error = np.dot(X, theta.T) - y
    cost = 1/(2*m) * np.dot(error.T, error)
    return cost, error
```

#### Gradient Descent

For gradient descent to work with multiple features, we have to do the same as in simple linear regression and update our theta values simultaneously over the amount of iterations and using the learning rate we supply. 

![alt text](https://www.dropbox.com/s/y6hjxzqenxz68ud/mvlrgradientdescent.png?raw=1 "gradient descent")

This is implemented in my code as:

```python
def gradient_descent(X, y, theta, alpha, iters):
    cost_array = np.zeros(iters)
    m = y.size
    for i in range(iters):
        cost, error = cost_function(X, y, theta)
        theta = theta - (alpha * (1/m) * np.dot(X.T, error))
        cost_array[i] = cost
    return theta, cost_array
```

This code also populltes a Numpy array of cost values so that we can plot a graph which shows the hopeful reduction in cost values as gradient descent runs.

#### A quick note on Feature Normalization

When working with multiple feature variables it will speed up gradient descent signicantly if they all are within a small range. We can achieve this using feature normalization. While there are a few ways to acheive this, a common used method uses the following method:

![alt text](https://www.dropbox.com/s/m2ylrpsz94ex51e/featurenormalization.png?raw=1 'feature normalization')

(The feature minus the mean of all the feature variables divided by the standard deviation)

## Implementation in Python

#### The Data

To implement this version of linear regression I decided to use a 2 feature hyptohetical dataset that featured the size of a house and how many bedrooms it had. The y value would be the price.

| Size | Bedrooms | Price  |
|------|----------|--------|
| 2104 | 3        | 399900 |
| 1600 | 3        | 329900 |
| 2400 | 3        | 369000 |
| 1416 | 2        | 232000 |

#### The Results

With theta values set to 0, the cost value returned the following amount:

```With initial theta values of [0. 0. 0.], cost error is 65591548106.45744```

After running gradient descent over the data, we see the cost value dropping:

![alt text](https://www.dropbox.com/s/c1zxdds7mh6zjgi/erroriterations.png?raw=1 "falling cost values")

After running for 2000 iterations (this value could be a lot smaller as convergence is acheived much earlier), gradient descent returns the following theta values:

```With final theta values of [340397.96353532 109848.00846026  -5866.45408497], cost error is 2043544218.7812893```

### Usage

`python housepricelinearregression.py`

## Links
* [Linear Descent from Scratch](https://www.kaggle.com/tentotheminus9/linear-regression-from-scratch-gradient-descent/notebook)

* [Linear Regression with Multiple Variables](https://www.coursera.org/learn/machine-learning/home/week/2)



