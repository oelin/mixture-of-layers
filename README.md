# Mixture of Layers

An implementation of Mixture of Layers (MoL) in PyTorch. We propose a method for neural networks to route information dynamically through their layers in an *arbitrary order*, allowing for in-context parameter tying.

![](https://i.ibb.co/XsMYr0c/mol.png)

## LayerRouter

The core of MoL is *LayerRouter*, a module that determines which layer the antecedent layer's activations should be forwarded through. Formally, LayerRouter is a function $f(\mathbf{x}_t, t)$ given by,

$$
    f(\mathbf{x}_t, t) = (g(\mathbf{x}_t, t), h(\mathbf{x}_t, t)),
$$ 

where $g(\mathbf{x}_t, t)$ returns a distribution over subsequent layer indices and $h(\mathbf{x}_t, t)$ is an arbitrary transformation on $\mathbf{x}_t$. The subsequent layer index is chosen as $\text{argmax}\, g(\mathbf{x}_t, t)$. Then, $h(\mathbf{x}_t, t)$ is given to it as input.
