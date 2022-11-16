## Overview

'Surpirsal' is the self-information: $-\ln p(\varphi)$
- $p(\varphi)$ is probability of observing some perticular snesory data $\varphi$ in a typical (habitable) env

Organisms cannot minimise surprisal directly, but an upper bound 'free energy', but maintains a probabilistic model of env (which includes their body), and attempt to minimise atypical event occurence in env measured by this model.
- model is (instantiated, parameterised) by physical varibles in organisms such as (neuronal activity, synaptic strengths).

Two key probability density
- Recognition density (R-density)
  - Organmism maintain an implicit representation of 'best guess' (probability distribution, like Bayesian belief) at the relevant variables that comprise env
  - update distribution to better reflect env (world), when receiving sensory signals
  - engage in an approximate-form process of Bayesian inference regarding env state, based on sensory observes
  - Though maybe more complex, assume agents approximate R-density as multivariate Guassian distribution
- Generative density (G-density)
  - Organmism needs implicit assumptions about how different env states shape sensory input
  - joint probability between sensory data and env variables
  - assume Gaussian
  - calculated as product of snesory input probability given some env state and a prior of organism current beliefs of env state probability distribution

Minimising free energy
- FE is calculated as K-L divergence between R- and G-density
- not directly measurable physical quantity
- to distinct from thermodynamic free energy, refer as 'variationalfree energy' (VFE), considering its role in variational Bayes
- Two fucntional consequences
  - provides an **upper bound on snesory surprisal**, organism estimates the dispersion of their constituent state and is central to life-long process interpretation of FEP
  - ideal Bayesian inference involves evaluating difficult integrals, organism implements variational Bayes, VFE minimising makes **R-density approximate to the posterior density of env variables given sensory data**

Action-perception cycle
- Minimising VFE by updating R-density can provide upper bound on surprisal but cannot minimise it directly
- organisms also act on their env to change sensory input, and thus minimise surprisal indirectly
- active inference mechanism, symmetric to perceptual inference (rather than infering cause of sensory data, infer actions that best make sensory data accord with internal representation of env)
- The action avoid organism constituent states dispersion is said to 'underpin' homoeostasis (homeorhesis, system stable about a complex trajectory of states rather than around a fixed point)
- equal to view actions as satifying hard constraits encoded in organism implicit env model
  - expectations (encode organism's 'desires'on env dynamics) in G-density cannot be met directly by perception and thus organism msut act to satify them
  - actions is more akin to control, behavior arises from a process of minimising deviation between organism's actual and desired trajectory
- other roles actions perform to disambiguate competing models

Two kinds of FEP-based researches
- central theory
  - particular explanation with Bayesian inference
- biologically plausible 'process theory'
  - how relevant probability densities could be parameterized by organism variables
  - how the variables should be expected to change to minimize VFE

Predictive coding
- the most commonly used implementation of FEP
- plausible mechanism for organism updating R-density given G-density 
- concept
  - deny classical notions of perception and cognition as a largely bottom-up process of evidence accumulation, or feature detection driven by conflict snesory signals
  - propose perceptual content is determined by top-down predictive signals arising from multi-level generative models of env causes of snesory signals, which are continually modified by bottom-up prediction error signals communicating mismatches between predicted and actual signals across hierarchical levels
- update R-density using a hierachical predictive coding
  - benefits (why to do this)
    - under suitable assumption VFE becomes formally equivalent to prediction error, which can readily be computed in brain 
    - provides a generic prior which allows high-level abstract sensory features extracted from data
    - sense in which brain models env can be conceptualized in a direct way as sensory signal prediction (suggest no need to know what env features to build the model of R- and G-density)