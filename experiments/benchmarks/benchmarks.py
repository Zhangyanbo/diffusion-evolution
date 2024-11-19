"""
This file contains basic functions for the benchmarks.

Main functions used in the benchmarks:

1. plot_background(obj:str, ax=None, title=None): 
   Plots the background for a given objective function.

2. get_obj(obj_name:str):
   Returns the objective function and its rescaled version.

3. get_cmap(obj_name:str):
   Returns the appropriate colormap for the given objective function.

"""
import matplotlib.pyplot as plt
from foobench import Objective
import torch
from color_plate import *


# set fitness target and distance scale to unify the scale and slope of the fitness
fitness_target = {
    "rosenbrock": 0,
    "beale": 0,
    "himmelblau": 0,
    "ackley": -12.5401,
    "rastrigin": -64.6249, # x_i = 3.51786
    "rastrigin_4d": -129.2498,
    "rastrigin_32d": -1033.9980,
    "rastrigin_256d": -8271.9844
}

distance_scale = {
    "rosenbrock": 287.51,
    "beale": 20,
    "himmelblau": 17.01,
    "ackley": 2,
    "rastrigin": 30,
    "rastrigin_4d": 60,
    "rastrigin_32d": 500,
    "rastrigin_256d": 4000
}


def visualize_2D(objective, ax=None, n_points=100, parameter_range=None, title=None, **imshow_kwargs):
    # get a list of points in the parameter range
    if parameter_range is None:
        parameter_range = [[-4, 4], [-4, 4]]
    xy_points = torch.meshgrid(*[torch.linspace(pr[0], pr[1], n_points) for pr in parameter_range])
    xy_points = torch.stack(xy_points, dim=-1).reshape(-1, len(parameter_range))
    Z = objective(xy_points)
    Z = Z.reshape(*[n_points for _ in parameter_range])

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    im = ax.imshow(torch.log(Z.T+1e-3), extent=(*parameter_range[0], *reversed(parameter_range[1])), **imshow_kwargs)
    ax.invert_yaxis()

    ax.set_title(title)

    return im

def get_cmap(obj_name:str):
    return custom_cmap

def rescale_wrapper(obj, vmin=None, vmax=None, **kwargs):
    if vmin is None or vmax is None:
        return obj
    def rescaled_obj(x):
        # return (obj(x) - vmin) / (vmax - vmin)
        return obj(x) - vmin
    return rescaled_obj

def inverse_wrapper(obj, eps=1e-2, p=2, **kwargs):
    def inverse_obj(x):
        return eps / (obj(x) ** p + eps)
    return inverse_obj

def objective_wrapper(obj, target=0, scale=1, eps=1e-3, p=2, **kwargs):
    def wrapped_obj(x):
        d = abs(obj(x) - target) / scale
        return eps / (d ** p + eps)
    return wrapped_obj

def energy_wrapper(obj, temperature=0.5, target=0, scale=1, **kwargs):
    def wrapped_obj(x):
        return torch.exp(-abs(obj(x) - target) / (temperature * scale))
    return wrapped_obj

def exp_wrapper(obj, temperature=1, **kwargs):
    def wrapped_obj(x):
        return torch.exp(obj(x) / temperature)
    return wrapped_obj

def _original_name(obj_name:str):
    if obj_name in ["rastrigin_4d", "rastrigin_32d", "rastrigin_256d"]:
        return "rastrigin"
    return obj_name

def get_obj(obj_name:str, eps=1e-2, target=None, scale=None, wrapper=None, **kwargs):
    if obj_name in ["rosenbrock", "beale", "himmelblau"]: # zero as the target
        obj = Objective(foo=obj_name, maximize=False, limit_val=100)
    else: # high values as the target
        obj = Objective(foo=_original_name(obj_name), maximize=True, limit_val=1e-9)
    
    if target is None:
        target = fitness_target[obj_name]
    if scale is None:
        scale = distance_scale[obj_name]
    
    if wrapper is None:
        wrapper = energy_wrapper
    return obj, wrapper(obj, target=target, scale=scale, eps=eps, **kwargs)

def get_visualize_obj(obj):
    return Objective(foo=obj.foo_name)

def plot_background(obj, ax=None, title=None):
    # obj = get_visualize_obj(obj)
    # _, obj = get_obj(obj)
    _, obj_rescaled = get_obj(obj.foo_name)
    cmap = get_cmap(obj.foo_name)
    visualize_2D(obj_rescaled, ax=ax, cmap=cmap, title=title)

    if ax is not None:
        # remove x, y label and ticks
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')