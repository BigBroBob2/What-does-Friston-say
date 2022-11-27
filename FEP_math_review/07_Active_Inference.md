## Active Inference

FEP framework:
- perceptual inference
  - minimise VFE by changing brain states to better predict sensory data
- active inference
  - minimise VFE by acts on world to alter sensory input to better fit sensory predictions, and thus changing snesations

Action does not appear explicitly in the formulation of VFE, but minimises VFE by changing sensory data. 

Brain must have an inverse model of how sensory data change with action:

$\varphi = \varphi (a)$

Where $\varphi$ is a single channel sensory data and $a$ is an action. Then we can have how the Laplace-encoded energy changes with respect to action:

$\frac{d E (\mu,\varphi)}{d a} = \frac{d \varphi}{d a} \frac{\partial E (\mu,\varphi)}{\partial \varphi}$

Thus we can write the same gradient decent scheme for action $a$:

$\dot{a} = -\kappa_{\alpha} \frac{d \varphi}{d a} \frac{\partial E (\mu,\varphi)}{\partial \varphi}$

Write for a vector of brain states in generalised coordinates:

$\dot{a} = -\kappa_{a} \sum_\alpha \frac{d \tilde{\varphi}_\alpha}{d a} \nabla_{\tilde{\varphi}_\alpha} E (\{\tilde{\mu}_\alpha\},\{\tilde{\varphi}_\alpha\})$

The idea for brain to innately possess inverse models is reasonable:
- excution of motor control can only depend on predictions about internal sensors (proprioceptors) by classic reflex arcs
- external and perhaps internal sensation are only indirectly minimised by action
- this provides an alternative to behavior optimization

### A simple agent-based model

Here is a model about a mobile agent must move to achieve desired local temperature $T_{\rm desire}$.

#### The world

Create a world (**generative process**), suppose a 1-D line and one simple heat source.

The agent's position on this plane is denoted as world variable $\vartheta$, and the agent's temperature depends on its position $\vartheta$:

$T(\vartheta) = \frac{T_0}{\vartheta^2 + 1}$

This equation gives the dynamics of the agent's world, the world causes of its sensory signals.

The corresponding gradient:

$\frac{d T}{d \vartheta} = -T_0 \frac{2\vartheta}{(\vartheta^2 + 1)^2}$

We allow the agent to sense the local temperature and the time derivative of the local temperature:

$\varphi = T + z_{\rm gp}$

$\varphi^{\prime} = \frac{d T}{d \vartheta} \vartheta^{\prime} + z_{\rm gp}^{\prime}$

Recall $z_{\rm gp}$ is the noise during reading the sensory data (rather than the brain model) described by the generative process.

We allow the agent to set its velocity as action $a$ to :

$\vartheta^{\prime} = a$

#### The agent's brain

The agent constructs a scheme to minimize VFE. 

Assume the agent has belief and knows the *priori* about the dynamics of world $f(\mu)$, and how sensory data is generated $g(\mu)$. 

The agent can learn these models, but we don't consider it here.

The agent has brain state $\mu$ which represents agent's estimation of the local temperature in the world. Consider the generative model up to the 3rd order:

$\mu^\prime  = f(\mu) + w$

$f(\mu) \equiv T_{\rm desire} - \mu$

$\mu^{\prime\prime}  = -\mu^\prime + w^\prime$

$\mu^{\prime\prime\prime}  = w^{\prime\prime}$

Recall the finite order by setting the highest order term to random fluctuation with large variance, which is the 3rd order here. The large variance can be effectively elimnimiated from the expression of Laplace-encoded energy.

Therefore we can build the agent belief model forms about its sensory data only to the 1st order (these are just model forms, not definitions):

$\varphi = g(\mu) + z$

$g(\mu) \equiv \mu$

$\varphi^\prime = \mu^\prime + z^\prime$

Which is similar to the the sensory data generation in the world generative process. This means the agent has belief and knows the *priori* about how sensory data is generated. 

Note that:
- the agent we built does not explicitly desire specific world states, there is no explicit *prior* on $\mu$
- the agent's generative model has a stable equilibrium point at $T_{\rm desire}$

The Laplace-encoded energy (ignoring the logarithm of variance as they play no role in minmizing with respect to $\mu$):

$E(\tilde{\mu},\tilde{\varphi}) = \frac{1}{2} \left ( \frac{\varepsilon_{z [0]}^{2}}{\sigma_{z [0]}} + \frac{\varepsilon_{z [1]}^{2}}{\sigma_{z [1]}} + \frac{\varepsilon_{w [0]}^{2}}{\sigma_{w [0]}} + \frac{\varepsilon_{w [1]}^{2}}{\sigma_{w [1]}} \right )$

Where the errors are defined as:

$\varepsilon_{z [0]} = \varphi-\mu$

$\varepsilon_{z [1]} = \varphi^\prime-\mu^\prime$

$\varepsilon_{w [0]} = \mu^\prime - g(\mu) = \mu^\prime + \mu - T_{\rm desire}$

$\varepsilon_{w [1]} = \mu^{\prime\prime} - g^{\prime} (\mu) = \mu^{\prime\prime} + \mu^\prime$

Note that $z,z^\prime$ and $w,w^\prime$ are noise terms in brain internal models representing noises of brain beliefs about world states and sensory data, which are different from $z_{\rm gp}$.

Use the gradient decent scheme:

$\dot{\mu} = \mu^\prime - \nabla_{\mu} E(\tilde{\mu},\tilde{\varphi}) = \mu^\prime - \kappa_{a} \left ( -\frac{\varepsilon_{z [0]}}{\sigma_{z [0]}} + \frac{\varepsilon_{w [0]}}{\sigma_{w [0]}} \right )$

$\dot{\mu}^\prime = \mu^{\prime\prime} - \nabla_{\mu^{\prime}} E(\tilde{\mu},\tilde{\varphi}) = \mu^{\prime\prime} - \kappa_{a} \left ( -\frac{\varepsilon_{z [1]}}{\sigma_{z [1]}} + \frac{\varepsilon_{w [0]}}{\sigma_{w [0]}} + \frac{\varepsilon_{w [1]}}{\sigma_{w [1]}} \right )$

$\dot{\mu}^{\prime\prime} = - \nabla_{\mu^{\prime\prime}} E(\tilde{\mu},\tilde{\varphi}) = - \kappa_{a} \frac{\varepsilon_{w [1]}}{\sigma_{w [1]}}$

Recall generative process:

$\vartheta^\prime = a$

$\varphi^{\prime} = \frac{d T}{d \vartheta} \vartheta^{\prime} + z_{\rm gp}^{\prime}$

The inverse model of action and sensory data:

$\frac{d \varphi^\prime}{d a} = \frac{d}{d a} (\frac{d T}{d \vartheta} a + z_{\rm gp}^{\prime}) = \frac{d T}{d \vartheta}$

Then we can write the gradient descent scheme:

$\dot{a} = -\kappa_\alpha \left ( \frac{d \varphi^\prime}{d a} \frac{\partial E}{\partial \varphi^\prime} \right ) = -\kappa_\alpha \frac{d T}{d \vartheta} \frac{\varepsilon_{z [1]}}{\sigma_{z [1]}}$

#### Simulation of the agent-world system

Steps:
1. simulate the world generative process by numerically integrating, including world states and agent's sensation;
2. implement preception by updating the brain states (constrained by this new sensory data) using the gradient decent scheme and numerically integrating;
3. action changes the world states and sensory data.

Results:
- absense of action:
  - small $\sigma_{z}$, large $\sigma_{w}$
    - the agent has higher confidence in sensory input than in internal model
    - ignores its internal model
    - successfully infers both the local temperature and its derivatives
    - gradient descent scheme is equivalent to least mean square estimation on the sensory data
  - euqally balance $\sigma_{z}$ and $\sigma_{w}$
    - VFE minimising cannot satisfy both sensory perception and predictions of internal models
    - what agent perceives is in conflict with what it desires
    - inferred local temperature is between the desired and sensed temperature
  - with action:
    - the agent does action to change the world state to bring in line with sensory predictions and the desired encoded within its dynamic model, the agent moves towards the desired temperatures
    - further reduction of surprisal by acting on the world, rather than optimizing the internal model