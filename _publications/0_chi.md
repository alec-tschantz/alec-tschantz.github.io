---
title: "Control as Hybrid Inference"
collection: publications
permalink: /publication/rl-inference
excerpt: 
date: 2020-07-11
venue: "Accepted @ Theoretical Foundations of RL Workshop (ICLR)"
paperurl: 'https://arxiv.org/abs/2007.05838'
---

__Alexander Tschantz__, _Beren Millidge, Anil K. Seth, Christopher L. Buckley_

[[Paper](https://arxiv.org/abs/2007.05838)] 

> The field of reinforcement learning can be split into model-based and model-free methods. Here, we unify these approaches by casting model-free policy optimisation as amortised variational inference, and model-based planning as iterative variational inference, within a `control as hybrid inference' (CHI) framework. We present an implementation of CHI which naturally mediates the balance between iterative and amortised inference. Using a didactic experiment, we demonstrate that the proposed algorithm operates in a model-based manner at the onset of learning, before converging to a model-free algorithm once sufficient data have been collected. We verify the scalability of our algorithm on a continuous control benchmark, demonstrating that it outperforms strong model-free and model-based baselines. CHI thus provides a principled framework for harnessing the sample efficiency of model-based planning while retaining the asymptotic performance of model-free policy optimisation.
