# Neural Langevin Equation for Molecular Dynamics Simulation

Coarse-grained (CG) molecular models behave differently from their all-atom counterparts due to the loss of degrees of freedom.

While there are advantages to CG models, a common issue is their tendency to exhibit faster diffusion compared to all-atom (AA) models.

Among various explanations, the most rigorous one relies on the Mori-Zwanzig projection operator theory. This theory introduces additional dissipative forces (friction) and random forces (thermal noise) on top of the conventional CG force field (conservative force).

$$
m\frac{d^2 r(t)}{dt^2} = f_C(r) - \int_0^t K(t - t') \frac{dr(t')}{dt'} dt' + R(t)
$$

where:
- $f_C(r)$ is the conservative force derived from the potential energy of the system.
- $K(t - t')$ is the time-dependent friction kernel.
- $R(t)$ is the colored and non-gaussian random force (thermal noise).

This can be simplified to the Langevin Equation with Markovian assumption, which, while losing some detailed information, is still effective at reproducing diffusion dynamics.

$$
m\frac{d^2 r(t)}{dt^2} = f_C(r) - \gamma v(t) + R(t)
$$

There are various theories and algorithms to determine Langevin parameters. However, many of them rely on microscopic simulation data, including Ab Initio MD simulation (AIMD). AIMD is one of the most accurate techniques but still another form of simulation with inherent assumptions.

Can we systematically learn or extract information from experimental data, the true source?

This is where the Neural Langevin Equation comes in. It first runs a simulation and then learns the conservative, dissipative, and random forces from the difference between the simulation itself and a reference dataset.

Note: The GLE version of this code will be made public soon. Stay tuned!
