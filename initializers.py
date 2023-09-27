import numpy as np
import deepPy.dense as dense
import deepPy.activations as activations
import deepPy.convolution as convolution


# For Dense Layer
def init_random(layer, dtype="float32"):
    if isinstance(layer, dense.Dense):
        layer.weights = np.random.randn(layer.input_shape, layer.output_shape).astype(dtype) * 0.01
        layer.bias = np.zeros((1, layer.output_shape)).astype(dtype)
    elif isinstance(layer, activations.PReLU):
        if isinstance(layer.input_shape, tuple):
            layer.alpha = np.random.rand(*layer.output_shape).astype(dtype) * 0.01
        else:
            layer.alpha = np.random.rand(layer.output_shape).astype(dtype) * 0.01
    elif isinstance(layer, convolution.Conv2D):
        layer.kernels = np.random.randn(*layer.kernel_shape).astype(dtype)
        layer.bias = np.random.randn(*layer.output_shape).astype(dtype)
    else:
        Exception("Layer Can't use Initializer")


# Recommended for ReLU
def init_Kaiming(layer, dtype="float32"):
    if isinstance(layer, dense.Dense):
        layer.weights = np.random.randn(layer.input_shape, layer.output_shape).astype(dtype) * np.sqrt(2 / layer.input_shape)
        layer.bias = np.zeros((1, layer.output_shape)).astype(dtype)
    # This initialization doesn't make sense for PReLu layer, so it is the same as random
    elif isinstance(layer, activations.PReLU):
        if isinstance(layer.input_shape, tuple):
            layer.alpha = np.random.rand(*layer.output_shape).astype(dtype) * 0.01
        else:
            layer.alpha = np.random.rand(layer.output_shape).astype(dtype) * 0.01
    elif isinstance(layer, convolution.Conv2D):
        layer.kernels = np.random.randn(*layer.kernel_shape).astype(dtype) * np.sqrt(2 / (layer.kernel_shape[1] * layer.kernel_shape[2] ** 2))
        layer.bias = np.random.randn(*layer.output_shape).astype(dtype)
    else:
        Exception("Layer Can't use Initializer")


# Recommended for Tanh
def init_Xavier(layer, dtype="float32"):
    if isinstance(layer, dense.Dense):
        layer.weights = np.random.randn(layer.input_shape, layer.output_shape).astype(dtype) * np.sqrt(1 / layer.input_shape)
        layer.bias = np.zeros((1, layer.output_shape)).astype(dtype)
    elif isinstance(layer, activations.PReLU):
        if isinstance(layer.input_shape, tuple):
            layer.alpha = np.random.rand(*layer.output_shape).astype(dtype) * 0.01
        else:
            layer.alpha = np.random.rand(layer.output_shape).astype(dtype) * 0.01
    elif isinstance(layer, convolution.Conv2D):
        layer.kernels = np.random.randn(*layer.kernel_shape).astype(dtype) * np.sqrt(1 / (layer.kernel_shape[1] * layer.kernel_shape[2] ** 2))
        layer.bias = np.random.randn(*layer.output_shape).astype(dtype)
    else:
        Exception("Layer Can't use Initializer")
