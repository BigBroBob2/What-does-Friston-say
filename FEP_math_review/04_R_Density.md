## R-Density: how brain encodes world states

Approximations that brain uses to explicitly instantiate R-density $q(\vartheta,\varphi)$ and specify VFE.

### Definition Table
Laplace-encoding:
|Symbol|Name|Description|
|:-|:-|:-|
|$N(\vartheta;\mu,\zeta)$     |Guassian assumption of R-density $q(\vartheta)$    |the Laplace approximation assumption of unknown R-density $q(\vartheta)$|
|$\mu$                        |Mean of R-density                                  |sufficient statistics parameter of R-density|
|$\zeta$                      |Variance of R-density                              |sufficient statistics parameter of R-density|
|$\zeta^*$                     |The optimal $\zeta$ that minimizes $F$             |analytically derived, removing explicit dependence of $F$ on $\zeta$|
|$p(\mu,\varphi)$             |Laplace-encoded G-density $p(\vartheta,\varphi)$                         ||
|$E(\mu,\varphi)$             |Laplace-encoded G-density $E(\vartheta,\varphi)$    ||

### Notes

Brain must explicitly encode R-density to implement methods talked about.

Suggested that neuronal quantities (e.g. neural activity) parametrise *sufficient statistics*
- encode a family of densities over world state $\vartheta$. 
- instantaneous brain state $\mu$ picks one density $q(\vartheta;\mu)$ from this family 
  - $\mu$ is parameter, not random variable

Find optimal $q(\vartheta;\mu)$ to minimize VFE is difficult. We need further approximation. 

### Two types of approximation
#### 1. R-density $q(\vartheta)$ can be factorized into independent sub-densities (variational filtering, Friston)
- $q(\vartheta) = q_{1}(\vartheta_{1})\dots q_{N}(\vartheta_{N})$
- the optimal $q(\vartheta;\mu)$ can be improved iteratively
- sub-densities affect each other only through mean-field quantities
  - see mean-field theory, MFT
    - $AB \approx \langle A \rangle B+A\langle B \rangle-\langle A \rangle \langle B \rangle$
  - yet MFT assumption may be too ideal and have not correct results
- refer to *ensemble learning*
  - unconstrained ensemble learning
  - not assume particular form of marginal density (Laplace approximation)

Recall $F$:

$F \equiv \int q(\vartheta) \ln \left ( \frac{q(\vartheta)}{p(\vartheta,\varphi)} \right ) d\vartheta$

World sub-states $\vartheta_\alpha, \alpha = 1,2,\dots,N$ must vary on distinctive time-scale: $\tau_1 < \tau_2 < \dots < \tau_N$. Factorisation approximation of $q(\vartheta)$:

$q(\vartheta) = \Pi_{\alpha=1}^{N} q_{\alpha}(\vartheta_\alpha)$

Normalization:

$\int q(\vartheta) d\vartheta = \Pi_{\alpha=1}^{N} \int q_\alpha(\vartheta_\alpha) d\vartheta_\alpha = 1$

$\int q_\alpha(\vartheta_\alpha) d\vartheta_\alpha = 1$

Substitute into $F$:

$F \equiv \int q(\vartheta) \ln \left ( \frac{q(\vartheta)}{p(\vartheta,\varphi)} \right ) d\vartheta = \int \Pi_{\alpha=1}^{N} q_{\alpha}(\vartheta_\alpha) \ln \left ( \frac{\Pi_{\alpha=1}^{N} q_{\alpha}(\vartheta_\alpha)}{p(\vartheta,\varphi)} \right ) d\vartheta = \int \Pi_{\alpha=1}^{N} q_{\alpha}(\vartheta_\alpha) \left ( \sum_{\sigma=1}^{N} \ln q_{\sigma}(\vartheta_\sigma) + E(\vartheta,\varphi) \right ) d\vartheta \equiv F(q(\vartheta);\varphi)$

Use normalization as constraints and write in Lagrange form:

$\lambda \left ( \Pi_{\alpha=1}^{N} \int q_\alpha(\vartheta_\alpha) d\vartheta_\alpha - 1 \right ) = 0$

Functional derivation of $q_{\beta},\beta = 1,2,\dots,N$:

$\delta_\beta F = \int \left ( \int \left ( \Pi_{\alpha=1,\alpha \neq \beta}^{N}  q_{\alpha}(\vartheta_\alpha) d\vartheta_\alpha \right ) \left ( \sum_{\sigma=1}^{N} \ln q_\sigma (\vartheta_\sigma) + E(\vartheta,\varphi) \right ) + 1 + \lambda \right ) d\vartheta_\beta \delta q_\beta$

Impose $\delta_\beta F \equiv 0$:

$\int \left ( \Pi_{\alpha=1,\alpha \neq \beta}^{N}  q_{\alpha}(\vartheta_\alpha) d\vartheta_\alpha \right ) \left ( \sum_{\sigma=1}^{N} \ln q_\sigma (\vartheta_\sigma) + E(\vartheta,\varphi) \right ) + 1 + \lambda = 0$

The optimal sub-density $q_\beta = q_\beta^*$:

$q_\beta^* = \exp \left ( -(1+\lambda) - \sum_{\sigma=1,\sigma \neq \beta}^{N} \int \Pi_{\alpha=1,\alpha \neq \beta}^{N}  q_{\alpha}(\vartheta_\alpha) \ln q_\sigma (\vartheta_\sigma) d\vartheta_\alpha - \int \Pi_{\alpha=1,\alpha \neq \beta}^{N}  q_{\alpha}(\vartheta_\alpha) E(\vartheta,\varphi) d\vartheta_\alpha \right )$

Define the partially-averaged energy $\varepsilon_\beta(\vartheta_\beta,\varphi)$:

$\varepsilon_\beta(\vartheta_\beta,\varphi) \equiv \int \Pi_{\alpha=1,\alpha \neq \beta}^{N}  q_{\alpha}(\vartheta_\alpha) E(\vartheta,\varphi) d\vartheta_\alpha$

The influence from other world sub-states $\{\vartheta_\alpha\}, \alpha \neq \beta$ occurs only through their average, which may be regarded as **mean-field approximation** of world states $\vartheta$.

Note we have the relation below, which means expectation of partially-averaged energy $\varepsilon_\beta(\vartheta_\beta,\varphi)$ under sub-density $q_\beta (\vartheta_\beta)$ is the average energy:

$\int q_\beta (\vartheta_\beta) \varepsilon_\beta(\vartheta_\beta,\varphi) d\vartheta_\beta = \int q(\vartheta) E(\vartheta,\varphi) d\vartheta$

Use normalization of $q_\beta(\vartheta_\beta)$ we have:

$\int q_\beta^* (\vartheta_\beta) d\vartheta_\beta = 1$

$e^{-(1+\lambda)} \exp \left ( - \sum_{\sigma=1,\sigma \neq \beta}^{N} \int \Pi_{\alpha=1,\alpha \neq \beta}^{N}  q_{\alpha}(\vartheta_\alpha) \ln q_\sigma (\vartheta_\sigma) d\vartheta_\alpha \right ) \int e^{-\varepsilon_\beta(\vartheta_\beta,\varphi)} d\vartheta_\beta = 1$

Where we solve $\lambda$, substitute it into $q_\beta^*$, similar to equilibrium canonical ensemble in statistical physics:

$q_\beta^* = \frac{e^{-\varepsilon_\beta(\vartheta_\beta,\varphi)}}{\int e^{-\varepsilon_\beta(\vartheta_\beta,\varphi)} d\vartheta_\beta}$

Define $Z_\beta$, similar to **partition function** of sub-system $\{\theta_\beta\}$:

$Z_\beta \equiv \int e^{-\varepsilon_\beta(\vartheta_\beta,\varphi)} d\vartheta_\beta$

Use the factorisation approximation:

$q^* (\vartheta) = \frac{e^{-\sum_{\alpha=1}^{N} \varepsilon_\alpha (\vartheta_\alpha,\varphi)}}{\Pi_{\alpha=1}^{N} \int e^{-\varepsilon_\alpha (\vartheta_\alpha,\varphi)} d\vartheta_\alpha} \equiv \frac{e^{-\varepsilon_T (\vartheta,\varphi)}}{Z_T}$

Where $Z_T$ may be called as 'total' partition function of world states $\vartheta$, and $\varepsilon_T$ the sum of partially-averaged energies.

The optimal R-density $q^* (\vartheta)$ should approximate posterior density $p(\vartheta \mid \varphi)$. However, because the partially-averaged energies contains R-density itself, to obtain a close form of $q^* (\vartheta)$ we should find a *self-consistent solution* (iterative) till convergence reaches.

recall $E(\vartheta,\varphi)$:

$E(\vartheta,\varphi) = -\ln p(\vartheta,\varphi)$

Use factorization approximation of G-density $p(\vartheta,\varphi)$:

$p(\vartheta,\varphi) = \Pi_{\sigma=1}^{N} p(\vartheta_\sigma,\varphi_\sigma) = \Pi_{\sigma=1}^{N} p(\vartheta_\sigma \mid \varphi_\sigma) p(\varphi_\sigma)$

We have:

$\varepsilon_\beta(\vartheta_\beta,\varphi) \equiv \int \Pi_{\alpha=1,\alpha \neq \beta}^{N}  q_{\alpha}(\vartheta_\alpha) E(\vartheta,\varphi) d\vartheta_\alpha = -\sum_{\sigma=1}^{N} \int \Pi_{\alpha=1,\alpha \neq \beta}^{N}  q_{\alpha}(\vartheta_\alpha) \ln p(\vartheta_\sigma,\varphi_\sigma) d\vartheta_\alpha = -\ln p(\vartheta_\beta,\varphi) - \sum_{\sigma=1,\sigma \neq \beta}^{N} \int \Pi_{\alpha=1,\alpha \neq \beta}^{N}  q_{\alpha}(\vartheta_\alpha) \ln p(\vartheta_\sigma,\varphi_\sigma) d\vartheta_\alpha$

recall $q_\beta^* (\vartheta_\beta)$:

$q_\beta^* = \frac{e^{-\varepsilon_\beta(\vartheta_\beta,\varphi)}}{\int e^{-\varepsilon_\beta(\vartheta_\beta,\varphi)} d\vartheta_\beta} = \frac{p(\vartheta_\beta,\varphi_\beta)}{\int p(\vartheta_\beta,\varphi_\beta) d\vartheta_\beta} = \frac{p(\vartheta_\beta,\varphi_\beta)}{p(\varphi_\beta)} \equiv p(\vartheta_\beta \mid \varphi_\beta)$

So we have the optimal $q^* (\vartheta)$:

$q^* (\vartheta) = \Pi_{\alpha=1}^{N} q_\alpha^* (\vartheta_\alpha) = \Pi_{\alpha=1}^{N} p(\vartheta_\alpha \mid \varphi_\alpha)$

Note that under factorization approximation of G-density $p(\vartheta,\varphi)$ we have exact Bayesian inference:

$q^* (\vartheta) = \Pi_{\alpha=1}^{N} p(\vartheta_\alpha \mid \varphi_\alpha) = p(\vartheta \mid \varphi)$

Otherwise we have approximate Bayesian inference. This optimal R-density $q^*$ selected from family of $q$ means the brain's solution to statistical inference posterior density about world states $\vartheta$ given sensory data $\varphi$.

So we have the minimized VFE $F^*$:

$F^* = \int q^* (\vartheta) \ln \left ( \frac{q^* (\vartheta)}{p(\vartheta,\varphi)} \right ) d\vartheta = \int q^* (\vartheta) \ln \left ( \frac{p(\vartheta \mid \varphi)}{p(\vartheta \mid \varphi) p(\varphi)} \right ) d\vartheta = -\ln p(\varphi)$

Note that sensory density $p(\varphi)$ is formally written as $p(\varphi \mid m)$, conditioned by agent $m$.

Recall upper bound of $F$:

$F \geq -\ln p(\varphi)$

So $F^* = -\ln p(\varphi)$ reaches the tight bound of surprisal.

#### 2. R-density is **Guassian** (Laplace approximation, also known as saddle point approximation in path integral, or partition function in statistical mechanics) (Generalized filtering, Friston)
R-density:

$q(\vartheta) \equiv N(\vartheta;\mu,\zeta) = \frac{1}{\sqrt{2\pi\zeta}}\exp \left ( -\frac{(\vartheta-\mu)^2}{2\zeta} \right ) = \frac{1}{Z}e^{-\varepsilon(\vartheta)}$

where $\mu$ and $\zeta$ are mean and variance of world state $\vartheta$:

$\mu = \int \vartheta q(\vartheta) d\vartheta$

$\zeta = \int (\vartheta - \mu)^2 q(\vartheta) d\vartheta$

It should be multivariate case, here examines univariate. Annotate:

$Z \equiv \sqrt{2\pi\zeta}$

$\varepsilon(\vartheta) \equiv \frac{(\vartheta-\mu)^2}{2\zeta}$

Recall VFE:

$F = \int q(\vartheta)E(\vartheta,\varphi) d\vartheta + \int q(\vartheta) \ln q(\vartheta) d\vartheta = \int q(\vartheta)E(\vartheta,\varphi) d\vartheta + \int q(\vartheta) \left ( -\ln Z - \varepsilon (\vartheta)\right ) d\vartheta = -\ln Z - \int q(\vartheta) \varepsilon (\vartheta) d\vartheta + \int q(\vartheta)E(\vartheta,\varphi) d\vartheta$

in which:

$-\ln Z = -\frac{1}{2}(\ln 2\pi\zeta)$

$- \int q(\vartheta) \varepsilon (\vartheta) d\vartheta = -\frac{1}{2}$

To simplify $\int q(\vartheta)E(\vartheta,\varphi) d\vartheta$, assume R-density $q(\vartheta)$ **dirac peaks** at $\vartheta = \mu$ and $E(\vartheta,\varphi)$ is smooth of $\vartheta$. Taylor expansion:

$\int q(\vartheta)E(\vartheta,\varphi) d\vartheta \approx \int q(\vartheta)\left (  E(\mu,\varphi) + \frac{\partial E}{\partial \vartheta} \mid_{\vartheta = \mu} (\vartheta-\mu) + \frac{1}{2} \frac{\partial^2 E}{\partial \vartheta^2} \mid_{\vartheta = \mu} (\vartheta-\mu)^2 \right ) d\vartheta = E(\mu,\varphi)+\frac{1}{2} \frac{\partial^2 E}{\partial \vartheta^2} \mid_{\vartheta = \mu} \zeta$

Therefore we have:

$F(\mu,\zeta,\varphi) = E(\mu,\varphi)+\frac{1}{2}\left ( \frac{\partial^2 E}{\partial \vartheta^2} \mid_{\vartheta = \mu} \zeta - \ln 2\pi\zeta - 1 \right )$

$\frac{\partial F}{\partial \zeta} = \frac{1}{2} \left ( \frac{\partial^2 E}{\partial \vartheta^2} \mid_{\vartheta = \mu} - \frac{1}{\zeta} \right )$

The optimal $\zeta = \zeta^*$:

$\zeta^* = \left ( \frac{\partial^2 E}{\partial \vartheta^2} \mid_{\vartheta = \mu} \right )^{-1}$

And we can write:

$F(\mu,\zeta^*,\varphi) = E(\mu,\varphi) - \frac{1}{2}\ln 2\pi\zeta^*$

In this form we have benefits:
- describe VFE of $p(\mu,\varphi)$ over sensory data $\varphi$ and R-density sufficient statistics $\mu$, rather than $p(\vartheta,\varphi)$ over unspecified world states $\vartheta$
- simplify expressions only depends on $\mu,\varphi$
- multivariate Guassian distribution under the more general assumption that world states $\vartheta$ co-vary weakly (both variance and covariance are small), then the full R-density $q(\vartheta)$ is still tightly peaked and Taylor expansion is still valid

For full multivariate case of $N$ variables, the Laplace-encoded energy:

$E(\{\mu_\alpha\},\{\varphi_\alpha\}) = -\ln p(\{\mu_\alpha\},\{\varphi_\alpha\}),\alpha = 1,2,\dots,N$

where we define $\{\mu_\alpha\}$ is vector of brain states, $\{\varphi_\alpha\}$ is vector of sensory data, corresponding to vector of world states $\{\vartheta_\alpha\}$. 

Minimising the Laplace-encoded energy suggests brain tries to represent only the maximum likelihood world causes (mean is max likelihood in Guassian) of sensory data $p(\{\mu_\alpha\},\{\varphi_\alpha\})$, but not essentially the details $p(\{\vartheta_\alpha\},\{\varphi_\alpha\})$.