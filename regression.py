import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

N_PARAMETERS = 10
LEARNING_RATE = 0.05
N_DATA = 9
show_basis = True

# input variable
x = np.random.sample(N_DATA) * 10
# target variable. The target variable consistis of the sin function plus gaussian noise.
t = np.sin(x) + np.random.normal(0, 0.2, N_DATA)

w = np.zeros(N_PARAMETERS)

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

def curv_mat(b):     
    dgtm = 0.00000001 # dont grow too much parameter
    return np.fromfunction(lambda i, j: dgtm * ((j**2-j)*(i**2-i)*b**(j+i-3)) / (j*i-2*i-2*j+4.9999), [N_PARAMETERS, N_PARAMETERS])

def gradient_dsc_step(basis):
    global w
    lambd = 0.0005 * reg_slider.val

    smotherizer = np.zeros(N_PARAMETERS)
    if basis == polinomial_basis:
        #print("0: \n", curv_mat(0.0001))
        #print("10: \n", w @ curv_mat(10))
        smotherizer = lambd * (w @ curv_mat(8))
       
    gradient = (t - (w @ basis(x))) @ np.transpose(basis(x))
    print(lambd * smotherizer)
    w += LEARNING_RATE * ( gradient - lambd * smotherizer )



# interface shit

fig, ax = plt.subplots()
plt.subplots_adjust(right=0.7, bottom=0.2)

ax.set_ylim(-2, 2)
ax.scatter(x * max_x, t * max_t)

x_plot = np.arange(0, 1, 0.01)
if(show_basis):
    basis_plots = []
    color_map = np.linspace(1,0,N_PARAMETERS)
    for i in range(N_PARAMETERS):
        plot, = ax.plot(x_plot * max_x, w[i] * basis(x_plot)[i], color=[color_map[i], 1, 1 - color_map[i]])
        basis_plots.append(plot)
curve_plot, = ax.plot(x_plot * max_x, (w @ basis(x_plot)) * max_t, color='red')

btn_axes = plt.axes([0.75, 0.5, 0.2, 0.06])
step_button = Button(btn_axes, 'step')

btn1000_axes = plt.axes([0.75, 0.4, 0.2, 0.06])
step1000_button = Button(btn1000_axes, '1000 steps')

slider_axes = plt.axes([0.2, 0.08, 0.65, 0.04])
reg_slider = Slider(slider_axes, 'regularization stenght', valmin=-1, valmax=20, valinit=1)

def change_basis_function(label):
    global basis_function
    global w

    basis_function = label
    w = np.ones(N_PARAMETERS)
    update_graph(0)

radio_axes = plt.axes([0.75, 0.63, 0.2, 0.2])
radio_axes.set_title('basis function')
radio_buttons = RadioButtons(radio_axes, ('polinomial', 'sine', 'gaussian'))
radio_buttons.on_clicked(change_basis_function)

def update_graph(steps):
    if(basis_function == 'polinomial'):
        basis = polinomial_basis
    if(basis_function == 'sine'):
        basis = sine_basis
    if(basis_function == 'gaussian'):
        basis = gaussian_basis

    for i in range(steps):
        gradient_dsc_step(basis)
    if(show_basis):
        for p in range(N_PARAMETERS):
            basis_plots[p].set_ydata(w[p] * basis(x_plot)[p])
    curve_plot.set_ydata((w @ basis(x_plot)) * max_t)
    plt.draw()

step_button.on_clicked(lambda event: update_graph(1))
step1000_button.on_clicked(lambda event: update_graph(1000))

plt.show()
