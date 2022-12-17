## Hierarchical generative model

How the brain could modify and learn G-density $p(\varphi,\vartheta)$.

The hierarchical generative model involves empirical *priors*.

Use gradient descent scheme to allow brain to infer parameters and hyperparameters of VFE and learn world dynamics based on sensory data.

### Hierarchical generative model
Definition table
|Symbol|Name \& Description|
|:-|:-|
|$\mu^{[i]}$        |brain states at cortex level $i, i=1,2,\dots,M$|
|$\mu^{[0]}$        |$\mu^{[0]} \equiv \varphi$, the sensory data is at the lowest cortical level|
|$g^{[i]}(\mu^{[i]})$   |Generative mapping functions of brain state $\mu^{[i]}$ to estimate lower level $\mu^{[i-1]}$|
|$z^{[i]}$          |Guassian noise when estimating $\mu^{[i]}$|
|$p \left ( \mu^{[M]} \right )$     |the highest level *prior*|

Assume the agent possess some knowledges or beliefs of *priori*, how the world works, in the form of a pre-specified generative model.
- However, FEP promises the ability to learn and infer arbitrary world dynamics.
- To achieve this, the brain starts out with a very generative hierachical genrative model of world dynamics, refined and moduled through experience. 


One benefit of using hierachical generative model is that the model avoids specifying explicit and fixed *prior* and thus can implement empirical Bayes.

There is fundamental distinction between inference and learning
- Inference means recognizing the current causes of sensory input by inferring time-variant latent world states
- This is different from inferring the time-invariant parameters that mediate dependencies among time-variant states

<!-- model -->

Key challenge for Bayesian inference models is how to specify *priors*.
- In hierachical models the higher layers provide empirical *priors*, or constraints on lower levels.
- Here the hierachical models are mapped onto the hierachical organisation of cortex.

Denote $\mu^{[i]}$ as brain states at cortex level $i, i=1,2,\dots,M$. The model can be written as:

$\mu^{[0]} \equiv \varphi = g^{[1]}(\mu^{[1]})+z^{[0]}$

$\mu^{[1]} = g^{[2]}(\mu^{[2]})+z^{[1]}$

$\mu^{[i]} = g^{[i+1]}(\mu^{[i+1]})+z^{[i]}$

$\mu^{[M]} = z^{[M]}, g^{[M+1]} \equiv 0$

Thus we have the specific G-density $p(\varphi,\mu)$ in the hierarchical model:

$p(\varphi,\mu) = p \left ( \varphi \mid \mu^{[1]}, \mu^{[2]},\dots,\mu^{[M]} \right ) p \left ( \mu^{[1]}, \mu^{[2]},\dots,\mu^{[M]} \right ) = p \left ( \mu^{[0]} \mid \mu^{[1]}, \mu^{[2]},\dots,\mu^{[M]} \right ) p \left ( \mu^{[1]}, \mu^{[2]},\dots,\mu^{[M]} \right )$

Assume Markovian properties of the transition probabilities form higher to lower levels:
$p(\varphi,\mu) = p \left ( \mu^{[0]} \mid \mu^{[1]} \right ) p \left ( \mu^{[0]} \mid \mu^{[1]} \right ) \dots p \left ( \mu^{[M-1]} \mid \mu^{[M]} \right ) p \left ( \mu^{[M]} \right )$

The *prior* at the highest level $p \left ( \mu^{[M]} \right )$ is Guassian:

$p \left ( \mu^{[M]} \right ) = \frac{1}{\sqrt{2\pi \sigma_{z}^{[M]}}} \exp \left ( -\frac{(\mu^{[M]})^2}{2\sigma_{z}^{[M]}} \right )$

Assume $z^{[i]},i=1,2,\dots,M$ are independent, we have:

$p \left ( \mu^{[i]} \mid \mu^{[i+1]} \right ) = \frac{1}{\sqrt{2\pi \sigma_{z}^{[i]}}} \exp \left ( -\frac{\left ( \mu^{[i]} - g^{[i+1]} (\mu^{[i+1]}) \right )^2}{2\sigma_{z}^{[i]}} \right )$

$p(\varphi,\mu) = \Pi_{i=0}^{M} \left ( \frac{1}{\sqrt{2\pi\sigma_{z}^{[i]}}} \right ) \exp \left ( -\sum_{i=0}^{M} \frac{\sigma_{z}^{[i]} ( \varepsilon^{[i+1]} )^2}{2} \right )$

Where $\varepsilon^{[i+1]}$ is the regularized *prediction error* when estimating lower level $\mu^{[i]}$ via $g^{[i+1]} (\mu^{[i+1]})$:

$\varepsilon^{[i+1]} \equiv \frac{\mu^{[i]} - g^{[i+1]} (\mu^{[i+1]})}{\sigma_{z}^{[i]}}$

Therefore we can write the Laplace-encoded energy as:

$E(\mu,\varphi) = \frac{1}{2} \left ( \sum_{i=0}^{M} \sigma_{z}^{[i]} ( \varepsilon^{[i+1]} )^2 + \sum_{i=0}^{M} \ln \sigma_{z}^{[i]} \right )$

Note that the noise at the top level of hierarchy is usually assumed to be large and thus approximately zero. This means the level below is effectively unconstrained (has no *prior*) and this type of inference is empirical Bayes.

### Full contruction of generative model (combining hierarchical and dynamical models)

Definition table
|Symbol|Name \& Description|
|:-|:-|
|$\tilde{\mu}_{\alpha}^{[i]}$       |$\tilde{\mu}_{\alpha}^{[i]} = (\mu_{\alpha [0]}^{[i]},\mu_{\alpha [1]}^{[i]},\dots)^T$, the generalized coordinates of brain state $\alpha$ in the cortical level $i$|
|$\tilde{\chi}_{\alpha}^{[i]}$      |the hidden state|
|$\tilde{\upsilon}_{\alpha}^{[i]}$      |the causal state|
|$\tilde{g}_{\alpha}^{[i]}$         |generative map to estimate the lower-level state $\tilde{\upsilon}_{\alpha}^{[i-1]}$|
|$\tilde{f}_{\alpha}^{[i]}$         |generative function to estimate the hidden state motion $D\tilde{\chi}_{\alpha}^{[i]}$|
|$\tilde{z}_{\alpha}^{[i]},\tilde{w}_{\alpha}^{[i]}$|Guassian noises|
|$p(\tilde{\chi}_\alpha^{[M]}, \tilde{\upsilon}_\alpha^{[M]})$|the *prior* density at the highest level $M$|
|$p(\tilde{\chi}_\alpha^{[i]} \mid \tilde{\upsilon}_\alpha^{[i]})$|the intra-level conditional probability of the hidden states $D\tilde{\chi}_{\alpha}^{[i]}$ conditioned in the causal state $\tilde{\upsilon}_{\alpha}^{[i]}$ via $\tilde{f}_{\alpha}^{[i]}$|
|$p(\tilde{\upsilon}_\alpha^{[i]} \mid \tilde{\chi}_\alpha^{[i+1]},\tilde{\upsilon}_\alpha^{[i+1]})$|likelihood density of the upper-level causal state $\tilde{\upsilon}_{\alpha}^{[i+1]}$ serves as *prior* for this-level causal states $\tilde{\upsilon}_{\alpha}^{[i]}$, the inter-level map between successive causal states|

Firstly, we note the brain state $\mu_\alpha$ can be divided into **hidden** states $\chi_\alpha$  and **causal** states $\upsilon_\alpha$ (directly observable):

$\mu_\alpha = (\chi_\alpha,\upsilon_\alpha)$

Remember that the lowest level of the hierarchical model brain states are sensory data, $\mu_{\alpha}^{[0]} \equiv \varphi_\alpha$, where the **hidden** states can be considered as **not directly observable** sensory data, and the **causal** states can be considered as **directly observable** sensory data.

Now we can generalize this distinction throughout every levels of the hierarchy. Each level is considered as **observing** only through the **causal** states of the lower level. Therefore we have:

$\tilde{\upsilon}_{\alpha}^{[i]} = \tilde{g}_{\alpha}^{[i+1]} (\tilde{\chi}_{\alpha}^{[i+1]},\tilde{\upsilon}_{\alpha}^{[i+1]}) + \tilde{z}_{\alpha}^{[i]}$

$D\tilde{\chi}_{\alpha}^{[i]} = \tilde{f}_{\alpha}^{[i]} (\tilde{\chi}_{\alpha}^{[i]},\tilde{\upsilon}_{\alpha}^{[i]}) + \tilde{w}_{\alpha}^{[i]}$

Where the first function means that the causal states of the one-level lower level $\tilde{\upsilon}_{\alpha}^{[i]}$ is estimated by this level's prior $\tilde{g}_{\alpha}^{[i+1]}$, based on this level's brain states $\tilde{\chi}_{\alpha}^{[i+1]}$ and $\tilde{\upsilon}_{\alpha}^{[i+1]}$; the second function means that the higher order of the hidden states $D\tilde{\chi}_{\alpha}^{[i]}$ is estimated in brain belief by $\tilde{f}_{\alpha}^{[i]}$.

Note that the lowest level of the causal states is the snesory data:

$\tilde{\upsilon}_{\alpha}^{[0]} = \tilde{\varphi}_\alpha$

The highest level $M$ of the causal states (*prior*) is large-variance fluctuations with zero means, but the generalized motions of the highest level hidden states are still present:

$\tilde{g}_{\alpha}^{[M+1]} \equiv 0$

$\tilde{\upsilon}_{\alpha}^{[M]} = \tilde{z}_{\alpha}^{[M]}$

The generalized coordinates of the causal states and hidden states are denoted as:

$\tilde{\upsilon}_{\alpha}^{[i]} \equiv ({\upsilon}_{\alpha [0]}^{[i]},{\upsilon}_{\alpha [1]}^{[i]}, \dots )^T$

$\tilde{\chi}_{\alpha}^{[i]} \equiv ({\chi}_{\alpha [0]}^{[i]},{\chi}_{\alpha [1]}^{[i]}, \dots )^T$

${\upsilon}_{\alpha [n]}^{[i]} \equiv \frac{d^n}{dt^n} {\upsilon}_{\alpha}^{[i]}$

${\chi}_{\alpha [n]}^{[i]} \equiv \frac{d^n}{dt^n} {\chi}_{\alpha}^{[i]}$

Use local-linearity assumption, we only have linear terms:

${g}_{\alpha [n]}^{[i+1]} ({\chi}_{\alpha [n]}^{[i+1]},{\upsilon}_{\alpha [n]}^{[i+1]}) \equiv \frac{\partial g}{\partial {\upsilon}_{\alpha [n]}^{[i+1]}} {\upsilon}_{\alpha [n]}^{[i+1]} \equiv {g}_{\alpha [n]}^{[i+1]}$

${f}_{\alpha [n]}^{[i+1]} ({\chi}_{\alpha [n]}^{[i]},{\upsilon}_{\alpha [n]}^{[i]}) \equiv \frac{\partial f}{\partial {\chi}_{\alpha [n]}^{[i]}} {\chi}_{\alpha [n]}^{[i]} \equiv {f}_{\alpha [n]}^{[i]}$

${g}_{\alpha [0]}^{[i+1]} ({\chi}_{\alpha [0]}^{[i+1]},{\upsilon}_{\alpha [0]}^{[i+1]}) = g({\chi}_{\alpha [0]}^{[i+1]},{\upsilon}_{\alpha [0]}^{[i+1]})$

${f}_{\alpha [0]}^{[i+1]} ({\chi}_{\alpha [0]}^{[i]},{\upsilon}_{\alpha [0]}^{[i]}) = f({\chi}_{\alpha [0]}^{[i]},{\upsilon}_{\alpha [0]}^{[i]})$

Then we have G-density:

$p(\tilde{\varphi},\tilde{\mu}) = \Pi_{\alpha=1}^{N} p(\tilde{\varphi}_\alpha,\tilde{\mu}_\alpha) = \Pi_{\alpha=1}^{N} \left ( p(\tilde{\mu}_\alpha^{[M]}) \Pi_{i=0}^{M-1} p(\tilde{\mu}_\alpha^{[i]} \mid \tilde{\mu}_\alpha^{[i+1]}) \right ) = \Pi_{\alpha=1}^{N} \left ( p(\tilde{\chi}_\alpha^{[M]},\tilde{\upsilon}_\alpha^{[M]}) \Pi_{i=0}^{M-1} p(\tilde{\chi}_\alpha^{[i]},\tilde{\upsilon}_\alpha^{[i]} \mid \tilde{\chi}_\alpha^{[i+1]},\tilde{\upsilon}_\alpha^{[i+1]}) \right ) = \Pi_{\alpha=1}^{N} \left ( p(\tilde{\chi}_\alpha^{[M]}, \tilde{\upsilon}_\alpha^{[M]}) \Pi_{i=0}^{M-1} p(\tilde{\chi}_\alpha^{[i]} \mid \tilde{\upsilon}_\alpha^{[i]}) p(\tilde{\upsilon}_\alpha^{[i]} \mid \tilde{\chi}_\alpha^{[i+1]},\tilde{\upsilon}_\alpha^{[i+1]}) \right )$

Here we need to note that only the causal states $\tilde{\upsilon}_\alpha^{[i]}$ are involved in the inter-level transitions in the hierarchy, which means:

$p(\tilde{\chi}_\alpha^{[i]} \mid \tilde{\upsilon}_\alpha^{[i]},\tilde{\chi}_\alpha^{[i+1]},\tilde{\upsilon}_\alpha^{[i+1]}) = p(\tilde{\chi}_\alpha^{[i]} \mid \tilde{\upsilon}_\alpha^{[i]})$

Also we have the following established fact:

$p(\tilde{\chi}_\alpha^{[0]} \mid \tilde{\upsilon}_\alpha^{[0]}) = 1$

About the intra-level conditional probability $p(\tilde{\chi}_\alpha^{[i]} \mid \tilde{\upsilon}_\alpha^{[i]})$, we assume correlation $\Sigma_{w_\alpha}^{[i]}$ among the generalized states at different dynamical orders:

$p(\tilde{\chi}_\alpha^{[i]} \mid \tilde{\upsilon}_\alpha^{[i]}) = \frac{1}{\sqrt{(2\pi)^{n_{\rm max}+1} | \Sigma_{w_{\alpha}}^{[i]} |}} \exp \left ( -\frac{1}{2}(D\tilde{\chi}_{\alpha}^{[i]}-\tilde{f}_{\alpha}^{[i]})^T (\Sigma_{w_\alpha}^{[i]})^{-1} (D\tilde{\chi}_{\alpha}^{[i]}-\tilde{f}_{\alpha}^{[i]}) \right )$

About the conditional probability $p(\tilde{\upsilon}_\alpha^{[i]} \mid \tilde{\chi}_\alpha^{[i+1]},\tilde{\upsilon}_\alpha^{[i+1]})$ linking successive causal states, we assume correlation $\Sigma_{z_\alpha}^{[i]}$ among the generalized states at different dynamical orders:

$p(\tilde{\upsilon}_\alpha^{[i]} \mid \tilde{\chi}_\alpha^{[i+1]},\tilde{\upsilon}_\alpha^{[i+1]}) = \frac{1}{\sqrt{(2\pi)^{n_{\rm max}+1} | \Sigma_{z_{\alpha}}^{[i]} |}} \exp \left ( -\frac{1}{2}(\tilde{\upsilon}_{\alpha}^{[i]}-\tilde{g}_{\alpha}^{[i+1]})^T (\Sigma_{z_\alpha}^{[i]})^{-1} (\tilde{\upsilon}_{\alpha}^{[i]}-\tilde{g}_{\alpha}^{[i+1]}) \right )$

About the *prior* density $p(\tilde{\chi}_\alpha^{[M]}, \tilde{\upsilon}_\alpha^{[M]})$ at the highest level, we have:

$p(\tilde{\chi}_\alpha^{[M]}, \tilde{\upsilon}_\alpha^{[M]}) = \left ( \frac{1}{\sqrt{(2\pi)^{n_{\rm max}+1} | \Sigma_{z_{\alpha}}^{[M]} |}} \exp \left ( -\frac{1}{2}(\tilde{\upsilon}_{\alpha}^{[M]})^T (\Sigma_{z_\alpha}^{[M]})^{-1} (\tilde{\upsilon}_{\alpha}^{[M]}) \right ) \right ) \left ( \frac{1}{\sqrt{(2\pi)^{n_{\rm max}+1} | \Sigma_{w_{\alpha}}^{[M]} |}} \exp \left ( -\frac{1}{2}(\tilde{\chi}_{\alpha}^{[M]} - \tilde{f}_{\alpha}^{[M]})^T (\Sigma_{w_\alpha}^{[M]})^{-1} (\tilde{\chi}_{\alpha}^{[M]} - \tilde{f}_{\alpha}^{[M]}) \right ) \right )$

Therefore we have the Laplace-encoded energy:

$E_{\alpha} (\tilde{\mu}_{\alpha},\tilde{\varphi}_{\alpha}) = \frac{1}{2} (\tilde{\upsilon}_{\alpha}^{[M]})^T (\Sigma_{z_\alpha}^{[M]})^{-1} (\tilde{\upsilon}_{\alpha}^{[M]}) + \frac{1}{2} \ln | \Sigma_{z_\alpha}^{[M]} | + \frac{1}{2} (D\tilde{\chi}_{\alpha}^{[M]} - \tilde{f}_{\alpha}^{[M]})^T (\Sigma_{w_\alpha}^{[M]})^{-1} (D\tilde{\chi}_{\alpha}^{[M]} - \tilde{f}_{\alpha}^{[M]}) + \frac{1}{2} \ln | \Sigma_{w_\alpha}^{[M]} | + \sum_{i=0}^{M-1} \left ( \frac{1}{2} (\tilde{\upsilon}_{\alpha}^{[i]}-\tilde{g}_{\alpha}^{[i+1]})^T (\Sigma_{z_\alpha}^{[i]})^{-1} (\tilde{\upsilon}_{\alpha}^{[i]}-\tilde{g}_{\alpha}^{[i+1]}) + \frac{1}{2} \ln | \Sigma_{z_\alpha}^{[i]} | + \frac{1}{2} (D\tilde{\chi}_{\alpha}^{[i]} - \tilde{f}_{\alpha}^{[i]})^T (\Sigma_{w_\alpha}^{[i]})^{-1} (D\tilde{\chi}_{\alpha}^{[i]} - \tilde{f}_{\alpha}^{[i]}) + \frac{1}{2} \ln | \Sigma_{w_\alpha}^{[i]} | \right )$

$E (\tilde{\mu},\tilde{\varphi}) = \sum_{\alpha=1}^{N} E_{\alpha} (\tilde{\mu}_{\alpha},\tilde{\varphi}_{\alpha})$

### The full construct recognition dynamics and neuronal activity

Now combine recognition dynamics with full construct. 

The gradient decrease scheme for the dynamical causal states:

$\dot{\upsilon}_{\alpha [n]}^{[i]} - D {\upsilon}_{\alpha [n]}^{[i]} = -\kappa_{z} \hat{\upsilon}_{\alpha [n]}^{[i]} \nabla_{\tilde{\upsilon}_{\alpha}} E(\tilde{\mu},\tilde{\varphi})$

Where we have (assume independent):

$\hat{\upsilon}_{\alpha [n]}^{[i]} \nabla_{\tilde{\upsilon}_{\alpha}} E(\tilde{\mu},\tilde{\varphi}) = \frac{\partial}{\partial \upsilon_{\alpha [n]}^{[i]}} \left ( \frac{1}{2} (\tilde{\upsilon}_{\alpha}^{[i-1]}-\tilde{g}_{\alpha}^{[i]})^T (\Sigma_{z_\alpha}^{[i-1]})^{-1} (\tilde{\upsilon}_{\alpha}^{[i-1]}-\tilde{g}_{\alpha}^{[i]}) + \frac{1}{2} (\tilde{\upsilon}_{\alpha}^{[i]}-\tilde{g}_{\alpha}^{[i+1]})^T (\Sigma_{z_\alpha}^{[i]})^{-1} (\tilde{\upsilon}_{\alpha}^{[i]}-\tilde{g}_{\alpha}^{[i+1]}) + \frac{1}{2} (\tilde{\chi}_{\alpha}^{[i]} - \tilde{f}_{\alpha}^{[i]})^T (\Sigma_{w_\alpha}^{[i]})^{-1} (\tilde{\chi}_{\alpha}^{[i]} - \tilde{f}_{\alpha}^{[i]}) \right )$

$\frac{\partial}{\partial \upsilon_{\alpha [n]}^{[i]}} \left ( (\tilde{\upsilon}_{\alpha}^{[i-1]}-\tilde{g}_{\alpha}^{[i]})^T (\Sigma_{z_\alpha}^{[i-1]})^{-1} (\tilde{\upsilon}_{\alpha}^{[i-1]}-\tilde{g}_{\alpha}^{[i]}) \right ) = 2 (\Sigma_{z_\alpha}^{[i-1]})^{-1} (\tilde{\upsilon}_{\alpha}^{[i-1]}-\tilde{g}_{\alpha}^{[i]}) \left ( - \frac{\partial \tilde{g}_{\alpha}^{[i]}}{\partial \upsilon_{\alpha [n]}^{[i]}} \right )$ 

$\frac{\partial}{\partial \upsilon_{\alpha [n]}^{[i]}} \left ( (\tilde{\upsilon}_{\alpha}^{[i]}-\tilde{g}_{\alpha}^{[i+1]})^T (\Sigma_{z_\alpha}^{[i]})^{-1} (\tilde{\upsilon}_{\alpha}^{[i]}-\tilde{g}_{\alpha}^{[i+1]}) \right ) = 2 (\Sigma_{z_\alpha}^{[i]})^{-1} (\tilde{\upsilon}_{\alpha}^{[i]}-\tilde{g}_{\alpha}^{[i+1]})$

$\frac{\partial}{\partial \upsilon_{\alpha [n]}^{[i]}} \left ( (\tilde{\chi}_{\alpha}^{[i]} - \tilde{f}_{\alpha}^{[i]})^T (\Sigma_{w_\alpha}^{[i]})^{-1} (\tilde{\chi}_{\alpha}^{[i]} - \tilde{f}_{\alpha}^{[i]}) \right )  = 2 (\Sigma_{w_\alpha}^{[i]})^{-1} (\tilde{\chi}_{\alpha}^{[i]} - \tilde{f}_{\alpha}^{[i]}) \left ( - \frac{\partial \tilde{f}_{\alpha}^{[i]}}{\partial \upsilon_{\alpha [n]}^{[i]}} \right )$

Define the *error units*, where $\sigma^{-1}$ can be called as *precisions*:

$\xi_{z,\alpha [n]}^{[i]} \equiv (\sigma_{z, \alpha [n]}^{[i-1]})^{-1} \left ( {\upsilon}_{\alpha [n]}^{[i-1]}-{g}_{\alpha [n]}^{[i]} (\chi_{\alpha [n]}^{[i]}, \upsilon_{\alpha [n]}^{[i]}) \right )$

$\xi_{w,\alpha [n]}^{[i]} \equiv (\sigma_{w, \alpha [n]}^{[i]})^{-1} \left ( {\chi}_{\alpha [n+1]}^{[i]}-{f}_{\alpha [n]}^{[i]} (\chi_{\alpha [n]}^{[i]}, \upsilon_{\alpha [n]}^{[i]}) \right )$

Thus ${\upsilon}_{\alpha [n]}^{[i]}$ and ${\chi}_{\alpha [n]}^{[i]}$ similarly represent *state units* (*representation units*) within neuronal populations. 
- In predictive coding (hierachical messages passing in cortical networks), the *error units* $\xi_{z,\alpha [n]}^{[i]}$ receive signals from lower-level causal states ${\upsilon}_{\alpha [n]}^{[i-1]}$, and also the same-level ${\upsilon}_{\alpha [n]}^{[i]}$ and ${\chi}_{\alpha [n]}^{[i]}$ via generative function ${g}_{\alpha [n]}^{[i]}$.
- The *error units* $\xi_{w,\alpha [n]}^{[i]}$ specify prediction-error in the within-level dynamics, between the objective hidden state ${\chi}_{\alpha [n+1]}^{[i]}$ and its estimation from lower-level causal and hidden state ${\upsilon}_{\alpha [n]}^{[i]}$ and ${\chi}_{\alpha [n]}^{[i]}$ via the different generative function ${f}_{\alpha [n]}^{[i]}$.

Therefore, we can write the dynamics of the causal state:

$\dot{\upsilon}_{\alpha [n]}^{[i]} = D\upsilon_{\alpha [n]}^{[i]} + \kappa_{z} \left ( \frac{\partial {g}_{\alpha [n]}^{[i]}}{\partial \upsilon_{\alpha [n]}^{[i]}} \xi_{z,\alpha [n]}^{[i]} - \xi_{z,\alpha [n]}^{[i+1]} + \frac{\partial {f}_{\alpha [n]}^{[i]}}{\partial \upsilon_{\alpha [n]}^{[i]}} \xi_{w,\alpha [n]}^{[i]} \right )$

Which shows that the hierarchical links are made among nearest neighbor cortical levels. The representation units of causal states $\upsilon_{\alpha [n]}^{[i]}$ are updated with higher-level error units $\xi_{z,\alpha [n]}^{[i+1]}$,  and the same-level error units $\xi_{z,\alpha [n]}^{[i]}$ and $\xi_{w,\alpha [n]}^{[i]}$.

Similarly, we can write the dynamics of the hidden state:

$\dot{\chi}_{\alpha [n]}^{[i]} = D{\chi}_{\alpha [n]}^{[i]} + \kappa_{w} \left ( - \xi_{w,\alpha [n-1]}^{[i]} + \frac{\partial {f}_{\alpha [n]}^{[i]}}{\partial \chi_{\alpha [n]}^{[i]}} \xi_{w,\alpha [n]}^{[i]} + \frac{\partial {g}_{\alpha [n]}^{[i]}}{\partial \chi_{\alpha [n]}^{[i]}} \xi_{z,\alpha [n]}^{[i]} \right )$

Which shows that the representation units ${\chi}_{\alpha [n]}^{[i]}$ are updated with lower-level error units $\xi_{w,\alpha [n-1]}^{[i]}$, and the same-level error units $\xi_{w,\alpha [n]}^{[i]}$ and $\xi_{z,\alpha [n]}^{[i]}$.

To summarize, prediction errors are passed up (bottom-up), and conditional expectations are passed down (top-down), consistent with predicticve coding.

### Parameters and hyperparameters: Synaptic efficacy and gain

We have shown that how world variables can be inferred by an appropriate G-density. We then discuss how the G-density can be learned. 

There are 3 timescales in the dynamics of neural systems:
- $\tau_{\mu}$ is the timescale of the fast dynamics of sufficient statistics of the encoded in R-density, $\mu \equiv (\chi,\upsilon)$.
- $\tau_{\theta}$ and $\tau_{\gamma}$ are the timescales of the slow dynamics of synaptic efficates $\theta$ and gains $\gamma$ which are parameterized implicitly in the laplace-encoded energy $E(\mu,\varphi)$ via the generative funtions $f$ and $g$, and the variances $\Sigma$.

$\tau_{\mu} < \tau_{\theta} < \tau_{\gamma}$

Slow variables are assumed to be approximately static (time-invariant) in contrast to time-varying neuronal states $\mu$. In other words, with respect to a small $\delta t$, changes in $\theta$ and $\gamma$ are much smaller than changes in $\mu$:

$\frac{\partial F}{\partial \theta} \frac{\delta \theta}{\delta t} \ll \frac{\partial F}{\partial \mu} \frac{\delta \mu}{\delta t}$

$\frac{\partial F}{\partial \gamma} \frac{\delta \gamma}{\delta t} \ll \frac{\partial F}{\partial \mu} \frac{\delta \mu}{\delta t}$

Recall VFE:

$F = \int q(\vartheta)E(\vartheta,\varphi) d\vartheta + \int q(\vartheta) \ln q(\vartheta) d\vartheta$

From the gradient decsent scheme, $\theta$ and $\gamma$ are not relevant to VFE but the integration of VFE over time, where the time-dependence of VFE is implicit through its arguments:

$S[F] \equiv \int F(\tilde{\mu},\tilde{\varphi};\theta,\gamma) d t$

Denote $\theta_{\alpha}^{[i]}$ as *parameters* and $\gamma_{\alpha}^{[i]}$ as *hyperparameters*, corresponding to brain state $\mu_{\alpha}^{[i]}$. Then *error units* can be generalized as:

$\xi_{z,\alpha [n]}^{[i]} \equiv \left ( \sigma_{z, \alpha [n]}^{[i-1]} (\gamma_{\alpha}^{[i-1]}) \right )^{-1} \left ( {\upsilon}_{\alpha [n]}^{[i-1]}-{g}_{\alpha [n]}^{[i]} (\chi_{\alpha [n]}^{[i]}, \upsilon_{\alpha [n]}^{[i]};\theta_{\alpha}^{[i]}) \right )$

$\xi_{w,\alpha [n]}^{[i]} \equiv \left ( \sigma_{w, \alpha [n]}^{[i]} (\gamma_{\alpha}^{[i]}) \right )^{-1} \left ( {\chi}_{\alpha [n+1]}^{[i]}-{f}_{\alpha [n]}^{[i]} (\chi_{\alpha [n]}^{[i]}, \upsilon_{\alpha [n]}^{[i]};\theta_{\alpha}^{[i]}) \right )$

We can write the recognition dynamics for the slow synaptic efficacy $\theta$ and the slow synaptic gain $\gamma$. Note that the gradient decsent schemeis applied using the integration of VFE $S[F]$. Assume a static model without dynamical orders:

$\dot{\theta}_{\alpha}^{[i]} = -\kappa_{\theta} \hat{\theta}_{\alpha}^{[i]} \nabla_{\theta_{\alpha}} S$

$\dot{\gamma}_{\alpha}^{[i]} = -\kappa_{\gamma} \hat{\gamma}_{\alpha}^{[i]} \nabla_{\gamma_{\alpha}} S$

And the second derivatives:

$\ddot{\theta}_{\alpha}^{[i]} = -\kappa_{\theta} \hat{\theta}_{\alpha}^{[i]} \nabla_{\theta_{\alpha}} E(\tilde{\mu},\tilde{\varphi};\theta,\gamma) = \kappa_{\theta} \sum_{n=0}^{n_{\rm max}} \left ( \frac{\partial {g}_{\alpha [n]}^{[i]}}{\partial \theta_{\alpha}^{[i]}} \xi_{z,\alpha [n]}^{[i]} + \frac{\partial {f}_{\alpha [n]}^{[i]}}{\partial \theta_{\alpha}^{[i]}} \xi_{w,\alpha [n]}^{[i]} \right )$

$\ddot{\gamma}_{\alpha}^{[i]} = -\kappa_{\gamma} \hat{\gamma}_{\alpha}^{[i]} \nabla_{\gamma_{\alpha}} E(\tilde{\mu},\tilde{\varphi};\theta,\gamma) = \kappa_{\gamma} \sum_{n=0}^{n_{\rm max}} \left ( -\frac{1}{2} \frac{\partial (\sigma_{z, \alpha [n]}^{[i]})^{-1}}{\partial \gamma_{\alpha}^{[i]}} (\sigma_{z, \alpha [n]}^{[i]})^{2} (\xi_{z,\alpha [n]}^{[i+1]})^{2} - \frac{\partial (\ln \sigma_{z, \alpha [n]}^{[i]})}{\partial \gamma_{\alpha}^{[i]}} -\frac{1}{2} \frac{\partial (\sigma_{w, \alpha [n]}^{[i]})^{-1}}{\partial \gamma_{\alpha}^{[i]}} (\sigma_{w, \alpha [n]}^{[i]})^{2} (\xi_{w,\alpha [n]}^{[i]})^{2} - \frac{\partial (\ln \sigma_{w, \alpha [n]}^{[i]})}{\partial \gamma_{\alpha}^{[i]}}  \right )$

To summarize, FEP prescribes *recognition dynamics* by gradient descent schemes on the sufficient statistics $\tilde{\mu}$, parameters $\theta$, and hyperparameters $\gamma$ on the Laplace-encoded energy $E(\tilde{\mu},\tilde{\varphi};\theta,\gamma)$, given the sensory input $\tilde{\varphi}$. At the end of this process, we get an optimal $\tilde{\mu}^{*}$ that represents the brain's posterior expectation of the environmental cause of the observed data:

$\tilde{\mu}^{*} = \arg \min_{\tilde{\mu}} F(\tilde{\mu},\tilde{\varphi})$

$F^{*} = F(\tilde{\mu}^{*},\tilde{\varphi})$

The only remaining task is to specify $f$ and $g$, which depend on the particular system being modelled. 

### Active inference on the full construct

VFE accounts for active inference by minimizing VFE with respect to action:

$a^{*} = \arg \min_{a} F(\tilde{\mu},\tilde{\varphi} (a))$

Where $a^{*}$ is the optimal action. Similarly we can write down the gradient descent scheme for the minimization in the full construct for action corresponding to brain's representation units $\mu_{\alpha}$:

$\dot{a}_{\alpha} = - \kappa_{a} \hat{a}_{\alpha} \nabla_{a_{\alpha}} E(\tilde{\mu},\tilde{\varphi} (a))$

Remember that the sensory input $\tilde{\varphi}$ is the lowest level of the causal states $\tilde{\upsilon}_{\alpha}^{[0]}$:

$\tilde{\varphi} = \tilde{\upsilon}_{\alpha}^{[0]}$

Therefore we have:

$\dot{a}_{\alpha} = -\kappa_{a} \sum_{n=0}^{n_{\rm max}} \frac{d \tilde{\varphi}_{\alpha[n]}}{d a_{\alpha}} \xi_{z, \alpha [n]}^{[1]}$

