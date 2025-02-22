"""Gradient Descent Assignment for CDS Intelligent Systems."""

import typing

import numpy as np
import random
from plotting import plot_grad_descent_1d, plot_linear_1d


# ============================================================================
# Example Hypothesis Functions
# ============================================================================


def linear_h(theta, x):
    """linear_h: The linear hypothesis regressor.

    :param theta: parameters for our linear regressor; shape (1, features)
    :type theta: np.ndarray
    :param x: input that model is predicting; shape (samples, features)
    :type x: np.ndarray
    :return: The predictions of our model on inputs X; shape (samples, 1)
    :rtype: np.ndarray
    """
    return (theta @ x.T).T


def linear_grad_h(theta, x):
    """linear_h: The gradient of the linear hypothesis regressor.

    :param theta: parameters for our linear regressor; shape (1, features)
    :type theta: np.ndarray
    :param x: input that model is predicting; shape (samples, features)
    :type x: np.ndarray
    :return: The gradient of our linear regressor; shape (samples, features)
    :rtype: np.ndarray
    """
    return x


def parabolic_h(theta, x):
    """parabolic_h: The parabolic hypothesis regressor.

    :param theta: parameters for our parabolic regressor; shape (1, features)
    :type theta: np.ndarray
    :param x: input that model is predicting; shape (samples, features)
    :type x: np.ndarray
    :return: The predictions of our model on inputs X; shape (samples, 1)
    :rtype: np.ndarray
    """
    return (theta @ (x ** 2).T).T


def parabolic_grad_h(theta, x):
    """parabolic_grad_h: The gradient of the parabolic hypothesis regressor.

    :param theta: parameters for our parabolic regressor; shape (1, features)
    :type theta: np.ndarray
    :param x: input that model is predicting; shape is (samples, features)
    :type x: np.ndarray
    :return: The gradient of our parabolic regressor; shape (samples, features)
    :rtype: np.ndarray
    """
    return x ** 2


# Add your own hypotheses if you want


def loss_f1(h, theta, x, y):
    """loss_f1 returns the loss for special function f1.

    This function is for demonstration purposes, since it ignores
    data points x and y.

    :param h: hypothesis function that is being used
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param theta: The parameters for our model, must be of shape (2,)
    :type theta: np.ndarray of shape (-1, 2)
    :param x: A matrix of samples and their respective features.
    :type x: np.ndarray of shape (samples, features)
    :param y: The expected targets our model is attempting to match
    :type y: np.ndarray of shape (samples,)
    :return: Return the function evaluation of theta, x, y
    :rtype: int or np.ndarray of shape (theta.shape[1],)
    """
    theta = np.reshape(theta, (-1, 2))
    w1 = theta[:, 0]
    w2 = theta[:, 0]
    return (
        -2 * np.exp(-((w1 - 1) * (w1 - 1) + w2 * w2) / 0.2)
        + -3 * np.exp(-((w1 + 1) * (w1 + 1) + y * y) / 0.2)
        + w1 * w1
        + w2 * w2
    )


def grad_loss_f1(h, grad_h, theta, x, y):
    """grad_loss_f1 returns the gradients for the loss of the f1 function.

    This function is for demonstration purposes, since it ignores
    data points x and y.

    :param h: The hypothesis function that predicts our output given weights
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: The gradient function of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param theta: The parameters for our model.
    :type theta: np.ndarray of shape (-1, 2)
    :param x: A matrix of samples and their respective features.
    :type x: np.ndarray of shape (samples, features)
    :param y: The expected targets our model is attempting to match
    :type y: np.ndarray of shape (samples,)
    :return: gradients for the loss function along the two axes
    :rtype: np.ndarray
    """
    theta = np.reshape(theta, (-1, 2))
    w1 = theta[:, 0]
    w2 = theta[:, 0]
    step = 1e-7
    grad_w1 = (loss_f1(w1 + step, w2) - loss_f1(w1, w2)) / step
    grad_w2 = (loss_f1(w1, w2 + step) - loss_f1(w1, y)) / step
    return np.array((grad_w1, grad_w2))


def l2_loss(
    h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta: np.ndarray,
    x, y):
    """l2_loss: standard l2 loss.

    The l2 loss is defined as (h(x) - y)^2. This is usually used for linear
    regression in the sum of squares.

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param theta: The parameters of our hypothesis function of shape (1, features)
    :type theta: np.ndarray
    :param x: Input matrix of shape (samples, features)
    :type x: np.ndarray
    :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    :type y: np.ndarray
    :return: The l2 loss value
    :rtype: float
    """
    return np.sum(np.square((h(theta, x) - y)))


def grad_l2_loss(h, grad_h, theta, x, y):
    """grad_l2_loss: The gradient of the standard l2 loss.

    The gradient of l2 loss is given by d/dx[(h(x) - y)^2] which is
    evaluated to 2*(h(x) - y)*h'(x).

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param theta: The parameters of our hypothesis fucntion
    :type theta: np.ndarray
    :param x: Input matrix of shape (samples, features)
    :type x: np.ndarray
    :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    :type y: np.ndarray
    :return: The l2 loss gradient of shape (1, features)
    :rtype: np.ndarray
    """
    return np.sum(2 * (h(theta, x) - y) * grad_h(theta, x), axis=0).reshape(1, -1)


# ============================================================================
# YOUR CODE GOES HERE:
# ============================================================================


def grad_descent(h, grad_h, loss_f, grad_loss_f, x, y, steps):
    """grad_descent: gradient descent algorithm on a hypothesis class.

    This does not use the matrix operations from numpy, this function
    uses the brute force calculations

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param loss_f: loss function that we will be optimizing on
    :type loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param grad_loss_f: the gradient of the loss function we are optimizing
    :type grad_loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param x: Input matrix of shape (samples, features)
    :type x: np.ndarray
    :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    :param y: np.ndarray
    :param steps: number of steps to take in the gradient descent algorithm
    :type steps: int
    :return: Ideal weights of shape (1, features), and the list of weights through time
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    theta = np.random.rand(1, x.shape[1])
    history = np.array([theta])
    for t in range(1, steps):
        grad = grad_loss_f(h, grad_h, theta, x, y)
        for i in range(0, len(theta)):
            theta[i] -= 0.001*grad[i]
        history = np.append(history, theta)

    return theta, history


def stochastic_grad_descent(h, grad_h, loss_f, grad_loss_f, x, y, steps):
    """grad_descent: gradient descent algorithm on a hypothesis class.

    This does not use the matrix operations from numpy, this function
    uses the brute force calculations

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param loss_f: loss function that we will be optimizing on
    :type loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param grad_loss_f: the gradient of the loss function we are optimizing
    :type grad_loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param x: Input matrix of shape (samples, features)
    :type x: np.ndarray
    :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    :param y: np.ndarray
    :param steps: number of steps to take in the gradient descent algorithm
    :type steps: int
    :return: Ideal weights of shape (1, features), and the list of weights through time
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    theta = np.random.rand(1, x.shape[1])
    history = np.array([theta])
    for t in range(1, steps):
        ix = random.randrange(len(x)-1)
        grad = grad_loss_f(h, grad_h, theta, x[ix], y[ix])
        for i in range(0, len(theta)):
            theta[i] -= .01*grad[i]
        history = np.append(history, theta)

    return theta, history


def minibatch_grad_descent(h, grad_h, loss_f, grad_loss_f, x, y, steps, batch_size=10):
    """grad_descent: gradient descent algorithm on a hypothesis class.

    This does not use the matrix operations from numpy, this function
    uses the brute force calculations

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param loss_f: loss function that we will be optimizing on
    :type loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param grad_loss_f: the gradient of the loss function we are optimizing
    :type grad_loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param x: Input matrix of shape (samples, features)
    :type x: np.ndarray
    :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    :param y: np.ndarray
    :param steps: number of steps to take in the gradient descent algorithm
    :type steps: int
    :return: Ideal weights of shape (1, features), and the list of weights through time
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    theta = np.random.rand(1, x.shape[1])
    history = np.array([theta])
    for t in range(1, steps):
        minibatch = [random.randrange(len(x)-1) for _ in range(batch_size)]
        grad = grad_loss_f(h, grad_h, theta, x[minibatch], y[minibatch])
        for i in range(0, len(theta)):
            theta[i] -= .01*grad[i]
        history = np.append(history, theta)

    return theta, history


def matrix_gd(h, grad_h, loss_f, grad_loss_f, x, y, steps):
    """grad_descent: gradient descent algorithm on a hypothesis class.

    This does not use the matrix operations from numpy, this function
    uses the brute force calculations

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param loss_f: loss function that we will be optimizing on
    :type loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param grad_loss_f: the gradient of the loss function we are optimizing
    :type grad_loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param x: Input matrix of shape (samples, features)
    :type x: np.ndarray
    :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    :param y: np.ndarray
    :param steps: number of steps to take in the gradient descent algorithm
    :type steps: int
    :param batch_size: number of elements in each training batch
    :type batch_size: int
    :return: Ideal weights of shape (1, features), and the list of weights through time
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    theta = np.random.rand(1, x.shape[1])
    history = np.array([theta])
    for t in range(1, steps):
        grad = grad_loss_f(h, grad_h, theta, x, y)
        theta -= 0.001*grad
        history = np.append(history, theta)

    return theta, history


def matrix_sgd(h, grad_h, loss_f, grad_loss_f, x, y, steps):
    """grad_descent: gradient descent algorithm on a hypothesis class.

    This does not use the matrix operations from numpy, this function
    uses the brute force calculations

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param loss_f: loss function that we will be optimizing on
    :type loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param grad_loss_f: the gradient of the loss function we are optimizing
    :type grad_loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param x: Input matrix of shape (samples, features)
    :type x: np.ndarray
    :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    :param y: np.ndarray
    :param steps: number of steps to take in the gradient descent algorithm
    :type steps: int
    :return: Ideal weights of shape (1, features), and the list of weights through time
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    theta = np.random.rand(1, x.shape[1])
    history = np.array([theta])
    for t in range(1, steps):
        ix = random.randrange(len(x)-1)
        grad = grad_loss_f(h, grad_h, theta, x[ix], y[ix])
        theta -= .01*grad
        history = np.append(history, theta)

    return theta, history


def matrix_minibatch_gd(h, grad_h, loss_f, grad_loss_f, x, y, steps, batch_size=10):
    """matrix_minibatch_gd: Mini-Batch GD using numpy matrix operations

    Stochastic Mini-batch GD with batches of size batch_size using numpy
    operations to speed up all of the operations

    :param h: hypothesis function that models our data (x) using theta
    :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param grad_h: function for the gradient of our hypothesis function
    :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param loss_f: loss function that we will be optimizing on
    :type loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param grad_loss_f: the gradient of the loss function we are optimizing
    :type grad_loss_f: typing.Callable[
        [
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray
        ],
        np.ndarray]
    :param x: Input matrix of shape (samples, features)
    :type x: np.ndarray
    :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    :param y: np.ndarray
    :param steps: number of steps to take in the gradient descent algorithm
    :type steps: int
    :param batch_size: number of elements in each training batch
    :type batch_size: int
    :return: Ideal weights of shape (1, features), and the list of weights through time
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    theta = np.random.rand(1, x.shape[1])
    history = np.array([theta])
    for t in range(1, steps):
        minibatch = [random.randrange(len(x)-1) for _ in range(batch_size)]
        grad = grad_loss_f(h, grad_h, theta, x[minibatch], y[minibatch])
        theta -= .01*grad
        history = np.append(history, theta)

    return theta, history


# ============================================================================
# Sample tests that you can run to ensure the basics are working
# ============================================================================

def save_linear_gif():
    """simple_linear: description."""
    x = np.arange(-3, 4, 0.1).reshape((-1, 1))
    y = 2*np.arange(-3, 4, 0.1).reshape((-1, 1))
    x_support = np.array((0, 4))
    y_support = np.array((-0.1, 200))
    plot_linear_1d(
        linear_h,
        linear_grad_h,
        l2_loss,
        grad_l2_loss,
        x,
        y,
        grad_descent,
        x_support,
        y_support
    )
    plot_grad_descent_1d(
        linear_h,
        linear_grad_h,
        l2_loss,
        grad_l2_loss,
        x,
        y,
        grad_descent,
        x_support,
        y_support
    )


def test_gd(grad_des_f):
    x = np.arange(-3, 4, 0.1).reshape((-1, 1))
    y = 2*np.arange(-3, 4, 0.1).reshape((-1, 1))
    x_support = np.array((0, 4))
    y_support = np.array((-0.1, 200))
    steps = 500

    theta, hist = grad_des_f(linear_h, linear_grad_h, l2_loss, grad_l2_loss, x, y, steps)
    return 1.99 < theta < 2.01


if __name__ == "__main__":
    results = np.array([])
    results = np.append(results, test_gd(grad_descent))
    results = np.append(results, test_gd(stochastic_grad_descent))
    results = np.append(results, test_gd(minibatch_grad_descent))
    results = np.append(results, test_gd(matrix_gd))
    results = np.append(results, test_gd(matrix_sgd))
    results = np.append(results, test_gd(matrix_minibatch_gd))

    print(results)
