using LinearAlgebra

function linear_h(theta, x)
    # linear_h: The linear hypothesis regressor.
    #
    # :param theta: parameters for our linear regressor; shape (1, features)
    # :type theta: np.ndarray
    # :param x: input that model is predicting; shape (samples, features)
    # :type x: np.ndarray
    # :return: The predictions of our model on inputs X; shape (samples, 1)
    # :rtype: np.ndarray
    #
    theta .⋅ x #Computes the dot product
end

function linear_grad_h(theta, x)
    # linear_h: The gradient of the linear hypothesis regressor.
    #
    # :param theta: parameters for our linear regressor; shape (1, features)
    # :type theta: np.ndarray
    # :param x: input that model is predicting; shape (samples, features)
    # :type x: np.ndarray
    # :return: The gradient of our linear regressor; shape (samples, features)
    # :rtype: np.ndarray
    #
    x
end

function parabolic_h(theta, x)
    # parabolic_h: The parabolic hypothesis regressor.
    #
    # :param theta: parameters for our parabolic regressor; shape (1, features)
    # :type theta: np.ndarray
    # :param x: input that model is predicting; shape (samples, features)
    # :type x: np.ndarray
    # :return: The predictions of our model on inputs X; shape (samples, 1)
    # :rtype: np.ndarray
    #
    theta ⋅ x.^2
end

function parabolic_grad_h(theta, x)
    # parabolic_grad_h: The gradient of the parabolic hypothesis regressor.
    #
    # :param theta: parameters for our parabolic regressor; shape (1, features)
    # :type theta: np.ndarray
    # :param x: input that model is predicting; shape is (samples, features)
    # :type x: np.ndarray
    # :return: The gradient of our parabolic regressor; shape (samples, features)
    # :rtype: np.ndarray
    #
    x.^2
end

# Add your own hypotheses if you want

function loss_f1(h, theta, x, y)
    # loss_f1 returns the loss for special function f1.
    #
    # This function is for demonstration purposes, since it ignores
    # data points x and y.
    #
    # :param h: hypothesis function that is being used
    # :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param theta: The parameters for our model, must be of shape (2,)
    # :type theta: np.ndarray of shape (-1, 2)
    # :param x: A matrix of samples and their respective features.
    # :type x: np.ndarray of shape (samples, features)
    # :param y: The expected targets our model is attempting to match
    # :type y: np.ndarray of shape (samples,)
    # :return: Return the function evaluation of theta, x, y
    # :rtype: int or np.ndarray of shape (theta.shape[1],)
    #
    theta = reshape(theta, :, 2)
    w1 = theta[:, 1]
    w2 = theta[:, 2]
    @. (
        -2 * exp(-((w1 - 1) * (w1 - 1) + w2 * w2) / 0.2)
        + -3 * exp(-((w1 + 1) * (w1 + 1) + y * y) / 0.2)
        + w1 * w1
        + w2 * w2
    )
end

#TODO: Does not work, Python implementation doesn't work either
function grad_loss_f1(h, grad_h, theta, x, y)
    # grad_loss_f1 returns the gradients for the loss of the f1 function.
    #
    # This function is for demonstration purposes, since it ignores
    # data points x and y.
    #
    # :param h: The hypothesis function that predicts our output given weights
    # :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param grad_h: The gradient function of our hypothesis function
    # :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param theta: The parameters for our model.
    # :type theta: np.ndarray of shape (-1, 2)
    # :param x: A matrix of samples and their respective features.
    # :type x: np.ndarray of shape (samples, features)
    # :param y: The expected targets our model is attempting to match
    # :type y: np.ndarray of shape (samples,)
    # :return: gradients for the loss function along the two axes
    # :rtype: np.ndarray
    #
    theta = reshape(theta, :, 2)
    w1 = theta[:, 1]
    w2 = theta[:, 2]
    step = 1e-7
    grad_w1 = (loss_f1(h, theta, w1 .+ step, w2) .- loss_f1(h, theta, w1, w2)) ./ step
    grad_w2 = (loss_f1(h, theta, w1, w2 .+ step) .- loss_f1(h, theta, w1, y)) ./ step
    hcat(grad_w1, grad_w2)
end

function l2_loss(h, grad_h, theta, x, y)
    # l2_loss: standard l2 loss.
    #
    # The l2 loss is defined as (h(x) - y)^2. This is usually used for linear
    # regression in the sum of squares.
    #
    # :param h: hypothesis function that models our data (x) using theta
    # :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param grad_h: function for the gradient of our hypothesis function
    # :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param theta: The parameters of our hypothesis function of shape (1, features)
    # :type theta: np.ndarray
    # :param x: Input matrix of shape (samples, features)
    # :type x: np.ndarray
    # :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    # :type y: np.ndarray
    # :return: The l2 loss value
    # :rtype: float
    #
    sum((h(theta, x) .- y).^2)
end

function grad_l2_loss(h, grad_h, theta, x, y)
    # grad_l2_loss: The gradient of the standard l2 loss.
    #
    # The gradient of l2 loss is given by d/dx[(h(x) - y)^2] which is
    # evaluated to 2*(h(x) - y)*h'(x).
    #
    # :param h: hypothesis function that models our data (x) using theta
    # :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param grad_h: function for the gradient of our hypothesis function
    # :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param theta: The parameters of our hypothesis fucntion
    # :type theta: np.ndarray
    # :param x: Input matrix of shape (samples, features)
    # :type x: np.ndarray
    # :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    # :type y: np.ndarray
    # :return: The l2 loss gradient of shape (1, features)
    # :rtype: np.ndarray
    #
    [sum(2 .* (h(theta, x) .- y) .* grad_h(theta, x))]
end

# ============================================================================
# YOUR CODE GOES HERE:
# ============================================================================


function grad_descent(h, grad_h, loss_f, grad_loss_f, x, y, steps)
    # grad_descent: gradient descent algorithm on a hypothesis class.
    #
    # This does not use the matrix operations from numpy, this function
    # uses the brute force calculations
    #
    # :param h: hypothesis function that models our data (x) using theta
    # :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param grad_h: function for the gradient of our hypothesis function
    # :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param loss_f: loss function that we will be optimizing on
    # :type loss_f: typing.Callable[
    #     [
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     np.ndarray,
    #     np.ndarray,
    #     np.ndarray
    #     ],
    #     np.ndarray]
    # :param grad_loss_f: the gradient of the loss function we are optimizing
    # :type grad_loss_f: typing.Callable[
    #     [
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     np.ndarray,
    #     np.ndarray,
    #     np.ndarray
    #     ],
    #     np.ndarray]
    # :param x: Input matrix of shape (samples, features)
    # :type x: np.ndarray
    # :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    # :param y: np.ndarray
    # :param steps: number of steps to take in the gradient descent algorithm
    # :type steps: int
    # :return: Ideal weights of shape (1, features), and the list of weights through time
    # :rtype: tuple[np.ndarray, np.ndarray]
    #
    theta = rand(Float64, 1, size(x)[2])
    history = [theta]
    for t in 1:steps
        grad = grad_loss_f(h, grad_h, theta, x, y)
        for i in 1:size(theta)[2]
            theta[i] -= 0.001*grad[i]
        end
        push!(history, theta)
    end
    theta, history
end

function stochastic_grad_descent(h, grad_h, loss_f, grad_loss_f, x, y, steps)
    # grad_descent: gradient descent algorithm on a hypothesis class.
    #
    # This does not use the matrix operations from numpy, this function
    # uses the brute force calculations
    #
    # :param h: hypothesis function that models our data (x) using theta
    # :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param grad_h: function for the gradient of our hypothesis function
    # :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param loss_f: loss function that we will be optimizing on
    # :type loss_f: typing.Callable[
    #     [
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     np.ndarray,
    #     np.ndarray,
    #     np.ndarray
    #     ],
    #     np.ndarray]
    # :param grad_loss_f: the gradient of the loss function we are optimizing
    # :type grad_loss_f: typing.Callable[
    #     [
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     np.ndarray,
    #     np.ndarray,
    #     np.ndarray
    #     ],
    #     np.ndarray]
    # :param x: Input matrix of shape (samples, features)
    # :type x: np.ndarray
    # :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    # :param y: np.ndarray
    # :param steps: number of steps to take in the gradient descent algorithm
    # :type steps: int
    # :return: Ideal weights of shape (1, features), and the list of weights through time
    # :rtype: tuple[np.ndarray, np.ndarray]
    #
    theta = rand(Float64, 1, size(x)[2])
    history = [theta]
    for t in 1:steps
        ix = rand((1:length(x)))
        grad = grad_loss_f(h, grad_h, theta, x[ix], y[ix])
        for i in 1:size(theta)[2]
            theta[i] -= .01*grad[i]
        end
        push!(history, theta)
    end
    theta, history
end

function minibatch_grad_descent(h, grad_h, loss_f, grad_loss_f, x, y, steps, batch_size=10)
    # grad_descent: gradient descent algorithm on a hypothesis class.
    #
    # This does not use the matrix operations from numpy, this function
    # uses the brute force calculations
    #
    # :param h: hypothesis function that models our data (x) using theta
    # :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param grad_h: function for the gradient of our hypothesis function
    # :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param loss_f: loss function that we will be optimizing on
    # :type loss_f: typing.Callable[
    #     [
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     np.ndarray,
    #     np.ndarray,
    #     np.ndarray
    #     ],
    #     np.ndarray]
    # :param grad_loss_f: the gradient of the loss function we are optimizing
    # :type grad_loss_f: typing.Callable[
    #     [
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     np.ndarray,
    #     np.ndarray,
    #     np.ndarray
    #     ],
    #     np.ndarray]
    # :param x: Input matrix of shape (samples, features)
    # :type x: np.ndarray
    # :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    # :param y: np.ndarray
    # :param steps: number of steps to take in the gradient descent algorithm
    # :type steps: int
    # :return: Ideal weights of shape (1, features), and the list of weights through time
    # :rtype: tuple[np.ndarray, np.ndarray]
    #
    theta = rand(Float64, 1, size(x)[2])
    history = [theta]
    for t in 1:steps
        minibatch = rand((1,length(x)), batch_size)
        grad = grad_loss_f(h, grad_h, theta, x[minibatch], y[minibatch])
        for i in 1:size(theta)[2]
            theta[i] -= .001*grad[i]
        end
        push!(history, theta)
    end
    theta, history
end

function matrix_gd(h, grad_h, loss_f, grad_loss_f, x, y, steps)
    # grad_descent: gradient descent algorithm on a hypothesis class.
    #
    # This does not use the matrix operations from numpy, this function
    # uses the brute force calculations
    #
    # :param h: hypothesis function that models our data (x) using theta
    # :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param grad_h: function for the gradient of our hypothesis function
    # :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param loss_f: loss function that we will be optimizing on
    # :type loss_f: typing.Callable[
    #     [
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     np.ndarray,
    #     np.ndarray,
    #     np.ndarray
    #     ],
    #     np.ndarray]
    # :param grad_loss_f: the gradient of the loss function we are optimizing
    # :type grad_loss_f: typing.Callable[
    #     [
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     np.ndarray,
    #     np.ndarray,
    #     np.ndarray
    #     ],
    #     np.ndarray]
    # :param x: Input matrix of shape (samples, features)
    # :type x: np.ndarray
    # :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    # :param y: np.ndarray
    # :param steps: number of steps to take in the gradient descent algorithm
    # :type steps: int
    # :return: Ideal weights of shape (1, features), and the list of weights through time
    # :rtype: tuple[np.ndarray, np.ndarray]
    #
    theta = rand(Float64, 1, size(x)[2])
    history = [theta]
    for t in 1:steps
        grad = grad_loss_f(h, grad_h, theta, x, y)
        theta .-= 0.001.*grad
        push!(history, theta)
    end
    theta, history
end

function matrix_sgd(h, grad_h, loss_f, grad_loss_f, x, y, steps)
    # grad_descent: gradient descent algorithm on a hypothesis class.
    #
    # This does not use the matrix operations from numpy, this function
    # uses the brute force calculations
    #
    # :param h: hypothesis function that models our data (x) using theta
    # :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param grad_h: function for the gradient of our hypothesis function
    # :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param loss_f: loss function that we will be optimizing on
    # :type loss_f: typing.Callable[
    #     [
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     np.ndarray,
    #     np.ndarray,
    #     np.ndarray
    #     ],
    #     np.ndarray]
    # :param grad_loss_f: the gradient of the loss function we are optimizing
    # :type grad_loss_f: typing.Callable[
    #     [
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     np.ndarray,
    #     np.ndarray,
    #     np.ndarray
    #     ],
    #     np.ndarray]
    # :param x: Input matrix of shape (samples, features)
    # :type x: np.ndarray
    # :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    # :param y: np.ndarray
    # :param steps: number of steps to take in the gradient descent algorithm
    # :type steps: int
    # :return: Ideal weights of shape (1, features), and the list of weights through time
    # :rtype: tuple[np.ndarray, np.ndarray]
    #
    theta = rand(Float64, 1, size(x)[2])
    history = [theta]
    for t in 1:steps
        ix = rand((1:length(x)))
        grad = grad_loss_f(h, grad_h, theta, x[ix], y[ix])
        theta .-= .01*grad
        push!(history, theta)
    end
    theta, history
end

function matrix_minibatch_gd(h, grad_h, loss_f, grad_loss_f, x, y, steps, batch_size=10)
    # grad_descent: gradient descent algorithm on a hypothesis class.
    #
    # This does not use the matrix operations from numpy, this function
    # uses the brute force calculations
    #
    # :param h: hypothesis function that models our data (x) using theta
    # :type h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param grad_h: function for the gradient of our hypothesis function
    # :type grad_h: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
    # :param loss_f: loss function that we will be optimizing on
    # :type loss_f: typing.Callable[
    #     [
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     np.ndarray,
    #     np.ndarray,
    #     np.ndarray
    #     ],
    #     np.ndarray]
    # :param grad_loss_f: the gradient of the loss function we are optimizing
    # :type grad_loss_f: typing.Callable[
    #     [
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    #     np.ndarray,
    #     np.ndarray,
    #     np.ndarray
    #     ],
    #     np.ndarray]
    # :param x: Input matrix of shape (samples, features)
    # :type x: np.ndarray
    # :param y: The expected targets our model is attempting to match, of shape (samples, 1)
    # :param y: np.ndarray
    # :param steps: number of steps to take in the gradient descent algorithm
    # :type steps: int
    # :return: Ideal weights of shape (1, features), and the list of weights through time
    # :rtype: tuple[np.ndarray, np.ndarray]
    #
    theta = rand(Float64, 1, size(x)[2])
    history = [theta]
    for t in 1:steps
        minibatch = rand((1,length(x)), batch_size)
        grad = grad_loss_f(h, grad_h, theta, x[minibatch], y[minibatch])
        theta .-= .001*grad
        push!(history, theta)
    end
    theta, history
end



# ============================================================================
# TESTING FUNCTIONS:
# ============================================================================
function test_gd(grad_des_f)
    x = reshape([-3:0.1:4 ...], :, 1)
    y = 2 .* x
    x_support = [0, 4]
    y_support = [-0.1, 200]
    steps = 500

    theta, hist = grad_des_f(linear_h, linear_grad_h, l2_loss, grad_l2_loss, x, y, steps)
    1.99 .< theta .< 2.01
end

macro test(func)
    append!(results, test_gd(@eval $func))
end


results = []
@test grad_descent
@test stochastic_grad_descent
@test minibatch_grad_descent
@test matrix_gd
@test matrix_sgd
@test matrix_minibatch_gd

results
