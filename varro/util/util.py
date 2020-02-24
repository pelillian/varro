"""
This module contains general utility functions.
"""

import os


def make_path(dir):
    """Ensures a given path exists."""
    os.makedirs(dir, exist_ok=True)


# TFBoard helper fns

def get_problem_range(problem):
    # Returns np.linspace(x_lower_bound, x_upper_bound, num_steps) for a given problem.__name__
    import numpy as np

    if problem == 'sinx':
        return np.linspace(-2*np.pi, 2*np.pi, 500)
    elif problem == 'cosx':
        return np.linspace(-2*np.pi, 2*np.pi, 500)
    elif problem == 'tanx':
        return np.linspace(-2*np.pi, 2*np.pi, 500)
    elif problem == 'x':
        return np.linspace(-10, 10, 500)
    elif problem == 'ras':
        return np.linspace(-5.12, 5.12, 0.01)
    elif problem == 'rosen':
        return np.linspace(-10, 10, 500)
    elif problem == 'step':
        return np.linspace(-10, 10, 500)
    elif problem == 'simple_step':
        return np.linspace(0, 1, 500)
    else:
        raise ValueError('Problem \'' + str(problem) + '\' not recognised')

def get_tb_fig(problem, y_pred):
    # Returns tf image object for a given problem.__name__ and np array of predictions
    import matplotlib.pyplot as plt
    import io
    import matplotlib.lines as mlines
    import numpy as np
    import tensorflow as tf

    range_min = np.min(get_problem_range(problem))
    range_max = np.max(get_problem_range(problem))
    X = np.linspace(range_min, range_max, 500)

    if problem == 'sinx':
        Y = np.sin(X)
    elif problem == 'cosx':
        Y = np.cos(X)
    elif problem == 'tanx':
        Y = np.tan(X)
    elif problem == 'x':
        Y = X
    elif problem == 'ras':
        Y = rastrigin(X)
    elif problem == 'rosen':
        Y = rosenbrock(X)
    elif problem == 'step':
        Y = (np.array(X) > 0).astype(float)
    elif problem == 'simple_step':
        Y = X
    # Convert plot to image
    figure = plt.figure(figsize=(12, 8))
    plt.plot(X,Y, color='b')
    plt.scatter(X,y_pred, color='g', alpha=0.25)
    plt.xlabel('X')
    plt.ylabel('Y')
    blue_line = mlines.Line2D([], [], color='b', marker='_',
                          markersize=15, label='ground truth')
    green_scatter = mlines.Line2D([], [], color='g', marker='o',
                          markersize=15, label='pred')
    plt.legend(handles=[blue_line, green_scatter])

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4, dtype=tf.uint8)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
