import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

N_PARAMETERS = 10
LEARNING_RATE = 0.04
N_DATA = 9

# input variable
x = np.random.sample(N_DATA) * 10
# target variable
t = np.sin(x) + np.random.normal(0, 0.1, N_DATA)

w = np.zeros(N_PARAMETERS)

def basis(x):
    return np.array(list(map(lambda i: x**i, np.arange(N_PARAMETERS))))

max_t = max(t) #normalizing the data
t /= max_t

max_x = max(x) #normalizing the data
x /= max_x

def gradient_dsc_step():
    global w
    gradient = (t - (w @ basis(x))) @ np.transpose(basis(x))
    w += LEARNING_RATE * gradient


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

ax.scatter(x * max_x, t * max_t)

x_plot = np.arange(0, 1, 0.01)
curve_plot, = ax.plot(x_plot * max_x, (w @ basis(x_plot)) * max_t, color='red')

btn_axes = plt.axes([0.8, 0.025, 0.1, 0.06])
step_button = Button(btn_axes, 'step')

btn1000_axes = plt.axes([0.52, 0.025, 0.2, 0.06])
step1000_button = Button(btn1000_axes, '1000 steps')

def update_graph(steps):
    for i in range(steps):
        gradient_dsc_step()
    curve_plot.set_ydata((w @ basis(x_plot)) * max_t)
    plt.draw()

step_button.on_clicked(lambda event: update_graph(1))
step1000_button.on_clicked(lambda event: update_graph(1000))

plt.show()
