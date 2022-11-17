## Core concepts of FEP
### Definition Table
VFE: 
|Symbol|Name|Description|
|:-|:-|:-|
|$\theta$                  |world state        |states of world (env and body), including well-defined characteristics (e.g. temperature) and joint unknown uncontrollable states evolving according to physical laws|
|$\varphi$                 |Sensory data       |world states as exogenous stimuli give rise to sensory inputs|
|$p(\theta,\varphi)$       |(Generative) G-density          |organism encodes prior beliefs about world states which cause corresponding sensory data|
|$p(\theta)$               |Prior density      |organism encodes prior beliefs about world states|
|$p(\varphi)$              |Sensory density    |organism encodes prior beliefs about sensory input, cannot be quantified given sensory data alone|
|$p(\theta \mid \varphi)$  |Posterior density  |inference that a perfect rational agent (with incomplete knowledge) would make about world state upon observing new sensory information, given organism prior assumptions|
|$p(\varphi \mid \theta)$  |Likelihood density |organism implicit beliefs about how world states map to sensory data|
|$q(\theta)$               |(Recognition) R-density         |organism (implicit) world state|


### Notes
Functional interface
- distinguish brain from world
- not necessarily a physical boundary
- many-to-one (non-bijective) mapping between ${\theta}$ and ${\varphi}$ is here
- brain, in conjuction with body, perform actions to modify sensory input

Goal of agent:
- important world states cannot be directly perceived, but can be inferred by Bayesian inference
- determine the probability of world states, given sensory input

G-density:

$p(\theta,\varphi) = p(\varphi \mid \theta) p(\theta)$

Given an observation (some particular data) $\varphi=\phi$, the posterior density:

$p(\theta \mid \varphi=\phi) = p(\theta \mid \phi) = \frac{p(\phi \mid \theta) p(\theta)}{p(\varphi=\phi)} = \frac{p(\phi \mid \theta) p(\theta)}{\int p(\phi \mid \theta)p(\theta) d\theta}$

Note we have normalization:

$\int \int p(\theta,\varphi) d\theta d\varphi = \int p(\theta) d\theta = \int p(\varphi) d\varphi = 1$

And marginal density:

$p(\theta) = \int p(\theta,\varphi) d\varphi$
$p(\varphi) = \int p(\theta,\varphi) d\theta$

Calculating marginal integral $\int p(\phi \mid \theta) p(\theta) d\theta$ is difficult. **Variational Bayes** is here used for (approximately) determining $p(\theta \mid \varphi)$. Introduce an auxilary density R-density $q(\theta)$, which has normalization:

$\int q(\theta) d\theta = 1$

Measure informarion-theortic divergence, K-L divergence:

$D_{KL} \left ( q(\theta \parallel p(\theta \mid \varphi)) \right ) = \int q(\theta) \ln \left (\frac{q(\theta)}{p(\theta \mid \varphi)} \right ) d\theta$

R-density $q(\theta)$ that minimizes this would provide good approximation to the true posterior density $p(\theta \mid \varphi)$. Rewrite K-L divergence as:

$D_{KL} \left ( q(\theta \parallel p(\theta \mid \varphi)) \right ) = \int q(\theta) \ln \left (\frac{q(\theta)}{p(\theta \mid \varphi)} \right ) d\theta = \int q(\theta) \ln \left ( \frac{q(\theta)}{p(\theta,\varphi)} \right ) d\theta + \ln p(\varphi) = F + \ln p(\varphi)$

Define VFE:

$F \equiv \int q(\theta) \ln \left ( \frac{q(\theta)}{p(\theta,\varphi)} \right ) d\theta$

VFE only depends on R-density (free to specify) and G-density. $\ln p(\varphi)$ only depends on sensory input $\varphi$ and independent of R-density. Therefore:

$\min F \leftrightarrow \min D_{KL} \left ( q(\theta \parallel p(\theta \mid \varphi)) \right )$

We know that K-L divergence is non-negative:

$D_{KL} \left ( q(\theta \parallel p(\theta \mid \varphi)) \right ) \geq 0$

So we have upper bound of $F$:

$F \geq -\ln p(\varphi)$

this is upper bound for surprisal. Only when R-density $q(\theta)$ becomes indentical with posterior density $p(\theta \mid \varphi)$ (i.e. $D_{KL} \left ( q(\theta \parallel p(\theta \mid \varphi)) \right ) = 0$), tight bound is reached.

FORMAL NOTE:
- agent's internal (implicit) density of sensory input $p(\varphi)$ should be formally written as $p(\varphi \mid m)$
  - this follows conventions in Bayesian statistics that an agent starts learning with arbitrary prior
  - $m$ is unwieldly and unchanged, so omit it

Analogs between information-theortic VFE and Helmholtz' thermodynamics free energy:

$F \equiv \int q(\theta) \ln \left ( \frac{q(\theta)}{p(\theta,\varphi)} \right ) d\theta = \int q(\theta)\left ( -\ln p(\theta,\varphi) \right ) d\theta + \int q(\theta) \ln q(\theta) d\theta = \int q(\theta)E(\theta,\varphi) d\theta + \int q(\theta) \ln q(\theta) d\theta$

The first term $\int q(\theta)\left ( -\ln p(\theta,\varphi) \right ) d\theta$ is averaging quantity $E(\theta,\varphi)$ over R-density $q(\theta)$ (called *average energy* in Helmholtz' thermodynamics free energy):

$E(\theta,\varphi) \equiv -\ln p(\theta,\varphi)$

And the second term $\int q(\theta) \ln q(\theta) d\theta$ is the *negative entropy* of R-density $q(\theta)$.



