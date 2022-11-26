## VFE minimization: how brain infers world states

Brain states change in such way that they implement **gradient descent** on VFE (recognition dynamics).

Let's say a brain state $\mu_{\alpha}$ is updated between sequential steps $t$ and $t+1$:

$\mu_{\alpha}^{t+1} = \mu_{\alpha}^{t} - \kappa \hat{\mu}_{\alpha} \cdot  \nabla_{\mu_{\alpha}} E(\{\mu_{\alpha}\},\{\varphi_{\alpha}\})$

Where $\kappa$ is learning rate, $\hat{\mu}_{\alpha}$ is unit vector along $\mu_{\alpha}$.

From discrete to continuous:

$\dot{\mu}_{\alpha} \equiv \mu_{\alpha}^{t+1} - \mu_{\alpha}^{t} = - \kappa \hat{\mu}_{\alpha} \cdot  \nabla_{\mu_{\alpha}} E(\{\mu_{\alpha}\},\{\varphi_{\alpha}\})$

To update dynamical orders of brain state $\mu_\alpha$:

$\mu_{\alpha [n]}^{t+1} = \mu_{\alpha [n]}^{t} - \kappa \hat{\mu}_{\alpha [n]} \cdot  \nabla_{\tilde{\mu}_{\alpha}} E(\{\tilde{\mu}_{\alpha}\},\{\tilde{\varphi}_{\alpha}\})$

However between dynamical orders we have:

$\mu_{\alpha [n]}^{t+1} - \mu_{\alpha [n]}^{t} = \mu_{\alpha [n+1]}^{t}$

It shows that it's not possible to make all orders zero, unable to construct a stationary solution at which $\nabla_{\tilde{\mu}_{\alpha}} E(\{\tilde{\mu}_{\alpha}\},\{\tilde{\varphi}_{\alpha}\}) = 0$.
- The motion of a point in generalized coordinate space $\dot{\tilde{\mu}}_\alpha$ (path of the mode, generalized velocity) does not have to be consistent with the brain encoded trajectory $D \tilde{\mu}_\alpha$ (mode of the path, average velocity), $D \tilde{\mu}_\alpha \neq \dot{\tilde{\mu}}_\alpha$
- Though, the two terms are the same under Hamilton's principle of stationary action, when "mode of the path" is not needed, $D \tilde{\mu}_\alpha \equiv 0$
- $D \tilde{\mu}_\alpha \equiv \left ( \mu_{\alpha [0]}^{\prime}, \mu_{\alpha [1]}^{\prime}, \dots \right ) \equiv \left ( \mu_{\alpha}^{\prime}, \mu_{\alpha}^{\prime\prime}, \dots \right )$
- $\dot{\tilde{\mu}}_\alpha = \frac{\partial}{\partial t} \tilde{\mu}_\alpha$
- Therefore the gradient descent scheme of is restated as (*not quite understand here, recall Kalman filter? expect-update*):
  - $\dot{\tilde{\mu}}_\alpha - D \tilde{\mu}_\alpha = - \kappa \hat{\mu}_{\alpha} \cdot  \nabla_{\mu_{\alpha}} E(\{\mu_{\alpha}\},\{\varphi_{\alpha}\})$ 