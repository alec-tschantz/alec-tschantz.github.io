---
title: "Learning action-oriented models through active inference"
collection: publications
permalink: /publication/plos-comp
excerpt: 
date: 2020-04-23
venue: 'PLOS Computational Biology'
paperurl: 'https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007805'
---

__Alexander Tschantz__, _Anil K. Seth, Christopher L. Buckley_

[[Paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007805)] [[Code](https://github.com/alec-tschantz/action-oriented)]

> Converging theories suggest that organisms learn and exploit probabilistic models of their environment. However, it remains unclear how such models can be learned in practice. The open-ended complexity of natural environments means that it is generally infeasible for organisms to model their environment comprehensively. Alternatively, action-oriented models attempt to encode a parsimonious representation of adaptive agent-environment interactions. One approach to learning action-oriented models is to learn online in the presence of goal-directed behaviours. This constrains an agent to behaviourally relevant trajectories, reducing the diversity of the data a model need account for. Unfortunately, this approach can cause models to prematurely converge to sub-optimal solutions, through a process we refer to as a bad-bootstrap. Here, we exploit the normative framework of active inference to show that efficient action-oriented models can be learned by balancing goal-oriented and epistemic (information-seeking) behaviours in a principled manner. We illustrate our approach using a simple agent-based model of bacterial chemotaxis. We first demonstrate that learning via goal-directed behaviour indeed constrains models to behaviorally relevant aspects of the environment, but that this approach is prone to sub-optimal convergence. We then demonstrate that epistemic behaviours facilitate the construction of accurate and comprehensive models, but that these models are not tailored to any specific behavioural niche and are therefore less efficient in their use of data. Finally, we show that active inference agents learn models that are parsimonious, tailored to action, and which avoid bad bootstraps and sub-optimal convergence. Critically, our results indicate that models learned through active inference can support adaptive behaviour in spite of, and indeed because of, their departure from veridical representations of the environment. Our approach provides a principled method for learning adaptive models from limited interactions with an environment, highlighting a route to sample efficient learning algorithms.



