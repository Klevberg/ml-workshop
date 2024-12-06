import matplotlib.pyplot as plt
import numpy as np


def plot_train_val_data(X_train, y_train, X_val, y_val):
    """
    Plots training and validation data side by side
    
    Args:
        X_train: Training features
        y_train: Training labels 
        X_val: Validation features
        y_val: Validation labels
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training data
    ax1.set_title("Training Data")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.scatter(X_train, y_train, color="green")

    # Plot validation data 
    ax2.set_title("Validation Data")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.scatter(X_val, y_val, color="orange")

    plt.tight_layout()
    plt.show()


def plot_linear_regression(X: np.ndarray, y: np.ndarray, weights: np.ndarray, title: str = "Data points and fitted linear function", scatter_color: str = "green"):
    """
    Plots the data points as a scatter plot and the fitted linear regression line
    
    Args:
        X: Input features
        y: Target values 
        weights: Regression weights [intercept, slope] where y = slope*x + intercept
        title: Plot title
        scatter_color: Color of the scatter plot
    """
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.scatter(X, y, alpha=0.5, color=scatter_color)
    x_values = np.array(plt.gca().get_xlim())
    ax.plot(x_values, weights[0] + weights[1]*x_values, color="red")


def plot_validation_results(learning_rates, validation_results):
    """
    Plots the validation MSE vs iterations for different learning rates.
    
    Args:
        learning_rates: List of learning rates used in the experiment.
        validation_results: Dictionary containing 'lr', 'iterations', and 'mse' lists.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for lr_val in learning_rates:
        mask = np.array(validation_results['lr']) == lr_val
        ax.plot(np.array(validation_results['iterations'])[mask],
                np.array(validation_results['mse'])[mask],
                'o-', label=f'lr={lr_val:.5f}')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Validation MSE')
    ax.set_title('Validation MSE vs Number of Iterations')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_regression_progression(X_train, y_train, X_val, y_val, gradient_descent, learning_rate=0.001):
    """
    Plot validation data and regression lines showing progression of gradient descent.
    
    Args:
        X_train: Training features
        y_train: Training labels 
        X_val: Validation features
        y_val: Validation labels
        gradient_descent: Gradient descent function
        learning_rate: Learning rate for gradient descent
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot validation data points
    ax.scatter(X_val, y_val, color='orange', alpha=0.75, label='Validation Data')

    x_line = np.array([min(X_val), max(X_val)])

    # Plot regression lines at different iterations with increasing opacity
    iterations_to_show = [0, 5, 10, 25, 100, 500, 1000, 1500]

    for i, iteration in enumerate(iterations_to_show):
        # Get weights for this iteration
        weights = gradient_descent(X_train, y_train, learning_rate=learning_rate, iterations=iteration)
        
        # Generate points for the line
        y_line = weights[0] + weights[1] * x_line
        
        # Plot line with increasing opacity
        alpha = 0.1 + (i * 0.9 / len(iterations_to_show))
        if iteration == 0:
            ax.plot(x_line, y_line, '-', color='green', alpha=0.5, label=f'Iteration {iteration}')
        else:
            ax.plot(x_line, y_line, '-', color='black', alpha=alpha, label=f'Iteration {iteration}')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Validation Data with Regression Line Progression\n0 to 1500 iterations, learning rate = {learning_rate}')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_loss_surface(X, y, calculate_loss_and_gradient, dim='2d', lr=0.05, n_iter=100, init_weights=[0, 0]):
    """
    Plots MSE loss function as contour (2d) or surface (3d) plot.
    Args:
        X, y: Data points
        gradient_loss_function, mse_loss: Functions to compute the gradient and loss
        dim: '2d' or '3d'
        lr: Learning rate
        n_iter: Number of iterations
        init_weights: Starting weights
    """
    if dim not in ['2d', '3d']:
        raise ValueError("dim must be '2d' or '3d'")
    
    # Create loss surface
    W0, W1 = np.meshgrid(np.linspace(-200, 200, 400), np.linspace(-100, 130, 230))
    loss = np.array([[calculate_loss_and_gradient(X, y, [w0, w1])[0] for w0, w1 in zip(row0, row1)] 
                     for row0, row1 in zip(W0, W1)])
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = ax if dim == '2d' else fig.add_subplot(111, projection='3d')
    ax.set_title(f"MSE Loss {'Contour' if dim == '2d' else 'Surface'}")
    ax.set_xlabel("Intercept"), ax.set_ylabel("Slope")
    
    # Plot loss surface
    if dim == '2d':
        surf = ax.contourf(W0, W1, loss, levels=500, cmap="coolwarm", alpha=0.8)
    else:
        surf = ax.plot_surface(W0, W1, loss, cmap='coolwarm', alpha=0.8)
        ax.set_zlabel("Loss")
    fig.colorbar(surf)
    
    # Plot gradient descent path
    weights = np.array(init_weights, dtype=float)
    path = [weights.copy()]
    for _ in range(n_iter):
        weights -= lr * calculate_loss_and_gradient(X, y, weights)[1]
        path.append(weights.copy())
    path = np.array(path)
    
    if dim == '2d':
        ax.plot(path[:, 0], path[:, 1], 'k.', markersize=1)
    else:
        ax.plot(path[:, 0], path[:, 1], [calculate_loss_and_gradient(X, y, w)[0] for w in path], 
               'k.', label='Gradient descent path')
        ax.legend()
    
    plt.tight_layout()
    plt.show()