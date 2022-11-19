## Core concepts of FEP
### Definition Table
VFE: 
|Symbol|Name|Description|
|:-|:-|:-|
|$\vartheta$                  |world state        |states of world (env and body), including well-defined characteristics (e.g. temperature) and joint unknown uncontrollable states evolving according to physical laws|
|$\varphi$                 |Sensory data       |world states as exogenous stimuli give rise to sensory inputs|
|$p(\vartheta,\varphi)$       |(Generative) G-density          |organism encodes prior beliefs about world states which cause corresponding sensory data|
|$p(\vartheta)$               |Prior density      |organism encodes prior beliefs about world states|
|$p(\varphi)$              |Sensory density    |organism encodes prior beliefs about sensory input, cannot be quantified given sensory data alone|
|$p(\vartheta \mid \varphi)$  |Posterior density  |inference that a perfect rational agent (with incomplete knowledge) would make about world state upon observing new sensory information, given organism prior assumptions|
|$p(\varphi \mid \vartheta)$  |Likelihood density |organism implicit beliefs about how world states map to sensory data|
|$q(\vartheta)$               |(Recognition) R-density         |organism (implicit) world state|


### Notes
Functional interface
- distinguish brain from world
- not necessarily a physical boundary
- many-to-one (non-bijective) mapping between ${\vartheta}$ and ${\varphi}$ is here
- brain, in conjuction with body, perform actions to modify sensory input

Goal of agent:
- important world states cannot be directly perceived, but can be inferred by Bayesian inference
- determine the probability of world states, given sensory input

G-density:

$p(\vartheta,\varphi) = p(\varphi \mid \vartheta) p(\vartheta)$

Given an observation (some particular data) $\varphi=\phi$, the posterior density:

$p(\vartheta \mid \varphi=\phi) = p(\vartheta \mid \phi) = \frac{p(\phi \mid \vartheta) p(\vartheta)}{p(\varphi=\phi)} = \frac{p(\phi \mid \vartheta) p(\vartheta)}{\int p(\phi \mid \vartheta)p(\vartheta) d\vartheta}$

Note we have normalization:

$\int \int p(\vartheta,\varphi) d\vartheta d\varphi = \int p(\vartheta) d\vartheta = \int p(\varphi) d\varphi = 1$

And marginal density:

$p(\vartheta) = \int p(\vartheta,\varphi) d\varphi$
$p(\varphi) = \int p(\vartheta,\varphi) d\vartheta$

Calculating marginal integral $\int p(\phi \mid \vartheta) p(\vartheta) d\vartheta$ is difficult. **Variational Bayes** is here used for (approximately) determining $p(\vartheta \mid \varphi)$. Introduce an auxilary density R-density $q(\vartheta)$, which has normalization:

$\int q(\vartheta) d\vartheta = 1$

Measure informarion-theortic divergence, K-L divergence:

$D_{KL} \left ( q(\vartheta \parallel p(\vartheta \mid \varphi)) \right ) = \int q(\vartheta) \ln \left (\frac{q(\vartheta)}{p(\vartheta \mid \varphi)} \right ) d\vartheta$

R-density $q(\vartheta)$ that minimizes this would provide good approximation to the true posterior density $p(\vartheta \mid \varphi)$. Rewrite K-L divergence as:

$D_{KL} \left ( q(\vartheta \parallel p(\vartheta \mid \varphi)) \right ) = \int q(\vartheta) \ln \left (\frac{q(\vartheta)}{p(\vartheta \mid \varphi)} \right ) d\vartheta = \int q(\vartheta) \ln \left ( \frac{q(\vartheta)}{p(\vartheta,\varphi)} \right ) d\vartheta + \ln p(\varphi) = F + \ln p(\varphi)$

Define VFE:

$F \equiv \int q(\vartheta) \ln \left ( \frac{q(\vartheta)}{p(\vartheta,\varphi)} \right ) d\vartheta$

VFE only depends on R-density (free to specify) and G-density. $\ln p(\varphi)$ only depends on sensory input $\varphi$ and independent of R-density. Therefore:

$\min F \leftrightarrow \min D_{KL} \left ( q(\vartheta \parallel p(\vartheta \mid \varphi)) \right )$

We know that K-L divergence is non-negative:

$D_{KL} \left ( q(\vartheta \parallel p(\vartheta \mid \varphi)) \right ) \geq 0$

So we have upper bound of $F$:

$F \geq -\ln p(\varphi)$

this is upper bound for surprisal. Only when R-density $q(\vartheta)$ becomes indentical with posterior density $p(\vartheta \mid \varphi)$ (i.e. $D_{KL} \left ( q(\vartheta \parallel p(\vartheta \mid \varphi)) \right ) = 0$), tight bound is reached.

FORMAL NOTE:
- agent's internal (implicit) density of sensory input $p(\varphi)$ should be formally written as $p(\varphi \mid m)$
  - this follows conventions in Bayesian statistics that an agent starts learning with arbitrary prior
  - $m$ is unwieldly and unchanged, so omit it

Analogs between information-theortic VFE and Helmholtz' thermodynamics free energy:

$F \equiv \int q(\vartheta) \ln \left ( \frac{q(\vartheta)}{p(\vartheta,\varphi)} \right ) d\vartheta = \int q(\vartheta)\left ( -\ln p(\vartheta,\varphi) \right ) d\vartheta + \int q(\vartheta) \ln q(\vartheta) d\vartheta = \int q(\vartheta)E(\vartheta,\varphi) d\vartheta + \int q(\vartheta) \ln q(\vartheta) d\vartheta$

The first term $\int q(\vartheta)\left ( -\ln p(\vartheta,\varphi) \right ) d\vartheta$ is averaging quantity $E(\vartheta,\varphi)$ over R-density $q(\vartheta)$ (called *average energy* in Helmholtz' thermodynamics free energy):

$E(\vartheta,\varphi) \equiv -\ln p(\vartheta,\varphi)$

And the second term $\int q(\vartheta) \ln q(\vartheta) d\vartheta$ is the *negative entropy* of R-density $q(\vartheta)$.



