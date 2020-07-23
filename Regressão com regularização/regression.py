import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

N_PARAMETERS = 10
N_DATA = 9
regularized = False
init = False
#regularization_strenght = 1

# input variable
x = np.random.sample(N_DATA) * 10
# target variable. The target variable consistis of the sin function plus gaussian noise.
t = np.sin(x) + np.random.normal(0, 0.1, N_DATA)

w = np.ones(N_PARAMETERS)

def polinomial_basis(x):
   return np.array(list(map(lambda i: x**i, np.arange(N_PARAMETERS))))

def sine_basis(x):
   return np.array(list(map(lambda i: np.sin(x * i), np.arange(N_PARAMETERS))))

def gaussian_basis(x):
    return np.array(list(map(lambda i: np.exp(-((x - i/10)**2)/0.02),  np.arange(N_PARAMETERS))))

basis_function = 'polinomial'
basis = polinomial_basis

max_t = max(t) #normalizing the data
t /= max_t

max_x = max(x) #normalizing the data
x /= max_x

def pseudo_inv_fit(basis):
    global w    
    phi = np.transpose(basis(x))
    phi_t = basis(x)
    if(regularized):
        # print(np.linalg.inv(regularization_strenght * np.identity(N_PARAMETERS) + (phi_t @ phi) ) @ phi_t @ t)
        w = np.linalg.inv(reg_slider.val * np.identity(N_PARAMETERS) + (phi_t @ phi) ) @ phi_t @ t
    else:
        w = np.linalg.pinv(phi) @ t




# interface shit

fig, ax = plt.subplots()
plt.subplots_adjust(right=0.7, bottom=0.2)

ax.set_ylim(-2, 2)
ax.scatter(x * max_x, t * max_t)

x_plot = np.arange(0, 1, 0.01)
curve_plot, = ax.plot(x_plot * max_x, (w @ basis(x_plot)) * max_t, color='red')

def change_basis_function(label):
    global basis_function
    global init
    global w

    init = False

    basis_function = label
    w = np.ones(N_PARAMETERS)
    update_graph(0)

def reg_button_click(label):
    global regularized
    regularized = True if label == "regularization" else False

def change_reg_strenght(val):
    if regularized and init:
        update_graph(1)

btn_axes = plt.axes([0.75, 0.22, 0.2, 0.06])
fit_button = Button(btn_axes, 'fit')

radio_axes = plt.axes([0.75, 0.63, 0.2, 0.2])
radio_axes.set_title('basis function')
radio_buttons = RadioButtons(radio_axes, ('polinomial', 'sine', 'gaussian'))
radio_buttons.on_clicked(change_basis_function)

reg_radio_axes = plt.axes([0.75, 0.4, 0.2, 0.15])
reg_button = RadioButtons(reg_radio_axes, ('no regularization', 'regularization'))
reg_button.on_clicked( reg_button_click  )

slider_axes = plt.axes([0.2, 0.08, 0.65, 0.04])
reg_slider = Slider(slider_axes, 'regularization stenght', valmin=0, valmax=2, valinit=1)

reg_slider.on_changed(change_reg_strenght)

def update_graph(steps):
    global init
    init = True

    if(basis_function == 'polinomial'):
        basis = polinomial_basis
    if(basis_function == 'sine'):
        basis = sine_basis
    if(basis_function == 'gaussian'):
        basis = gaussian_basis

    for i in range(steps):
        pseudo_inv_fit(basis)
    
    curve_plot.set_ydata((w @ basis(x_plot)) * max_t)
    plt.draw()

fit_button.on_clicked(lambda event: update_graph(1))

plt.show()
