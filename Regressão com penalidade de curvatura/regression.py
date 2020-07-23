import math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

N_PARAMETERS = 20
N_DATA = 10
show_basis = True
regularized = False
init = False
regularization_strenght = 0.1
reg_type = 'regularization'
integration_interval = (0, 10)

# input variable
x = np.random.sample(N_DATA) * 10
# target variable. The target variable consistis of the sin function plus gaussian noise.
t = 4 + np.sin(x) + np.random.normal(0, 0.2, N_DATA)

w = np.ones(N_PARAMETERS)

def polynomial_basis(x):
    """ returns a matrix with the applications of the polynomial basis functions to all data points in x"""
    return np.array(list(map(lambda i: x**i, np.arange(N_PARAMETERS))))

def sine_basis(x):
    """ returns a matrix with the applications of the sine basis functions to all data points in x"""
    return np.array(list(map(lambda i: np.sin(x * i), np.arange(N_PARAMETERS))))

def gaussian_basis(x):
    """ returns a matrix with the applications of the gaussian basis functions to all data points in x"""
    return np.array(list(map(lambda i: np.exp(-200*((x - i)**2)),  np.linspace(0, 1, N_PARAMETERS))))

basis_function = 'polynomial'
basis = polynomial_basis

max_t = max(t) #normalizing the data
t /= max_t

max_x = max(x) #normalizing the data
x /= max_x


def generate_curvature_matrix(a, b):
    """ genetes the matrix that contains the information needed to compute the influence each parameter have in the final curvature
        of the model between points a and b. The process to generate this matrix involves integrating, for some of the basis 
        functions i have computed the analytical form of the integral beforehand; for other basis functions, like the gaussian,
        no such analytical form exists, and a numerical method is needed. """

    if basis_function == 'polynomial':
        return 1/100 * np.fromfunction(lambda i, j: (j**2-j)*(i**2-i)*8**(i+j-3) / np.maximum(i+j-3, 1), [N_PARAMETERS, N_PARAMETERS])
    if basis_function == 'gaussian':
        gaussian_curvature = lambda i, j, x: (400*(x-i)**2-1)*(400*(x-j)**2-1)*np.exp(-200*(x-i)**2-200*(x-j)**2)        
        curvature_matrix = np.zeros([N_PARAMETERS, N_PARAMETERS])
        x = np.linspace(a, b, 400)
        index_possition = np.linspace(0, 1, N_PARAMETERS)
        for i in range(N_PARAMETERS):
            for j in range(N_PARAMETERS):
                # integral of the curvature
                curvature_matrix[i][j] = np.trapz( gaussian_curvature(index_possition[i], index_possition[j], x), x)
        return curvature_matrix

phi_line = generate_curvature_matrix(integration_interval[0], integration_interval[1])

def pseudo_inv_fit(basis):
    global w    
    phi = np.transpose(basis(x))
    phi_t = basis(x)

    print('real regularization strenght: ', regularization_strenght)
    
    if(regularized):
        if(reg_type == 'regularization'):
            w = np.linalg.inv(regularization_strenght * np.identity(N_PARAMETERS) + (phi_t @ phi) ) @ phi_t @ t
        else:
            w = np.linalg.inv(phi_t @ phi + regularization_strenght * phi_line ) @ phi_t @ t
    else:
        w = np.linalg.pinv(phi) @ t


# interface shit

fig, ax = plt.subplots()
plt.subplots_adjust(right=0.7, bottom=0.2)

ax.set_ylim(-2, 8)
ax.scatter(x * max_x, t * max_t)

x_plot = np.arange(0, 1, 0.01)
if(show_basis):
    basis_plots = []
    color_map = np.linspace(1,0,N_PARAMETERS)
    for i in range(N_PARAMETERS):
        plot, = ax.plot(x_plot * max_x, w[i] * basis(x_plot)[i], color=[color_map[i], 1, 1 - color_map[i]])
        basis_plots.append(plot)
curve_plot, = ax.plot(x_plot * max_x, (w @ basis(x_plot)) * max_t, color='red')

def change_basis_function(label):
    global basis_function
    global init
    global phi_line
    global w

    init = False

    basis_function = label
    w = np.ones(N_PARAMETERS)
    phi_line = generate_curvature_matrix(integration_interval[0], integration_interval[1])
    
    update_graph(0)

def reg_button_click(label):
    global regularized
    regularized = True if label == "regularization" else False

def change_reg_strenght(val):
    global regularization_strenght
    regularization_strenght = (np.exp(val) - 1) / 40000
    if regularized and init:
        update_graph(1)

def change_reg_type(label):
    global reg_type
    reg_type = label

btn_axes = plt.axes([0.75, 0.20, 0.2, 0.06])
fit_button = Button(btn_axes, 'fit')

radio_axes = plt.axes([0.75, 0.68, 0.2, 0.2])
radio_axes.set_title('basis function')
radio_buttons = RadioButtons(radio_axes, ('polynomial', 'gaussian'))
radio_buttons.on_clicked(change_basis_function)

reg_radio_axes = plt.axes([0.75, 0.5, 0.2, 0.15])
reg_button = RadioButtons(reg_radio_axes, ('no regularization', 'regularization'))
reg_button.on_clicked( reg_button_click  )

reg_type_axes = plt.axes([0.75, 0.32, 0.2, 0.15])
reg_type_button = RadioButtons(reg_type_axes, ('regularization', 'curvature penalty'))
reg_type_button.on_clicked(change_reg_type)

slider_axes = plt.axes([0.2, 0.08, 0.65, 0.04])
reg_slider = Slider(slider_axes, 'regularization stenght', valmin=0, valmax=40, valinit=6)

reg_slider.on_changed(change_reg_strenght)

def update_graph(steps):
    global init
    init = True

    if(basis_function == 'polynomial'):
        basis = polynomial_basis
    if(basis_function == 'sine'):
        basis = sine_basis
    if(basis_function == 'gaussian'):
        basis = gaussian_basis

    for i in range(steps):
        pseudo_inv_fit(basis)
    
    if(show_basis):
        for p in range(N_PARAMETERS):
            basis_plots[p].set_ydata(w[p] * basis(x_plot)[p])
    curve_plot.set_ydata((w @ basis(x_plot)) * max_t)
    plt.draw()

fit_button.on_clicked(lambda event: update_graph(1))

plt.show()