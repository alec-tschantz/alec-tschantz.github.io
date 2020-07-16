---
title: "Predictive Coding Approximates Backprop along Arbitrary Computation Graphs"
collection: publications
permalink: /publication/pc
excerpt: 
date: 2020-06-07
venue: 'ArXiv'
paperurl: 'https://arxiv.org/abs/2006.04182'
---

_Beren Millidge,_ __Alexander Tschantz__, _Christopher L Buckley_

[[Paper](https://arxiv.org/abs/2006.04182)] [[Code](https://github.com/BerenMillidge/PredictiveCodingBackprop)]

> Backpropagation of error (backprop) is a powerful algorithm for training machine learning architectures through end-to-end differentiation. However, backprop is often criticised for lacking biological plausibility. Recently, it has been shown that backprop in multilayer-perceptrons (MLPs) can be approximated using predictive coding, a biologically-plausible process theory of cortical computation which relies only on local and Hebbian updates. The power of backprop, however, lies not in its instantiation in MLPs, but rather in the concept of automatic differentiation which allows for the optimisation of any differentiable program expressed as a computation graph. Here, we demonstrate that predictive coding converges asymptotically (and in practice rapidly) to exact backprop gradients on arbitrary computation graphs using only local learning rules. We apply this result to develop a straightforward strategy to translate core machine learning architectures into their predictive coding equivalents. We construct predictive coding CNNs, RNNs, and the more complex LSTMs, which include a non-layer-like branching internal graph structure and multiplicative interactions. Our models perform equivalently to backprop on challenging machine learning benchmarks, while utilising only local and (mostly) Hebbian plasticity. Our method raises the potential that standard machine learning algorithms could in principle be directly implemented in neural circuitry, and may also contribute to the development of completely distributed neuromorphic architectures.
