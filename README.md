# FLAME: Adaptive Mixture-of-Experts for Continual Multimodal Multi-Task Learning

# Update for NeurIPS 2026 Rebuttal

## A. Expert Spectral Analysis

Cumulative spectral energy captured by the top-K components of each expert, measured at
three points in the continual-learning curriculum. Every figure compares three views over
10 layers and 5 experts (`fc1`/`fc2` per expert): the **input spectrum** (eigenvalues of the
input covariance `Cx`), the **weight-only** spectrum (Frobenius-normalized singular values
of the expert weights), and the **data-aware** spectrum (singular values of the test-time
activations). Dashed and dotted lines mark the 90% and 99% energy thresholds.

### Stage 0 — `pheno-density`

![Expert input/weight/data-aware spectra, stage 0 (pheno-density)](figs/stage0_pheno-density_expert_input_spectrum_comparison.png)

### Stage 1 — `los-birads-mortality`

![Expert input/weight/data-aware spectra, stage 1 (los-birads-mortality)](figs/stage1_los-birads-mortality_expert_input_spectrum_comparison.png)

### Stage 2 — `ihm-risk-readmission`

![Expert input/weight/data-aware spectra, stage 2 (ihm-risk-readmission)](figs/stage2_ihm-risk-readmission_expert_input_spectrum_comparison.png)

## B. Proofs for the Rebuttal Response

This section gives the full arguments behind the three claims in our response to
Reviewer `vmMT`. Notation follows the paper. At stage $j$, $W_i^{(j)}$ is the
rank-$r_j$ component reserved for expert $i$ and $\Pi_i$ the frozen stack of
reserved components. Task $t$ carries cursor $\tau = k(t)$ and its effective
expert weight is

$$W_i^{\mathrm{eff}}(\tau) = W_i^{(0)} + \sum_{j=1}^{\tau} W_i^{(j)}.$$

We write $f_t^{(s)}$ for the predictor of task $t$ after stage $s$ has
completed, $R_t$ and $\widehat{R}_t$ for its population and empirical risks
under $\mathcal{D}_t$, and $S_t(x)$ for the top-$K$ expert support selected for
sample $x$.

### (1) Forgetting is exactly zero

**Conditions.** (a) *Frozen task state*: after stage $\tau = k(t)$, every
parameter and state variable used by the forward pass of task $t$ is frozen.
This covers all expert and encoder components with index $j \le \tau$, the
stored router heads $\{G_m^{(\tau)}\}_{m \in \mathcal{M}_t}$, the task head
$h^{(t)}$, and any normalization or preprocessing statistics used by the task.
(b) *Cursor inference*: at inference, task identity determines $\tau$, the
stored router heads and the task head, and every compressed stack is evaluated
only through components with index $j \le \tau$.

Both are enforced by the training and inference protocol of App. E and Eq. (4).

**Claim.** For every stage $s \ge \tau$ and every input $x$,
$f_t^{(s)}(x) = f_t^{(\tau)}(x)$. Hence
$R_t(f_t^{(s)}) = R_t(f_t^{(\tau)})$, and any forgetting measure defined as
deterioration relative to task $t$'s post-stage-$\tau$ performance is exactly
zero.

**Proof.** Fix $s > \tau$; the case $s = \tau$ is trivial. We check each of the
four objects that the forward pass of task $t$ depends on.

*Expert weights.* Stage $s$ trains $\widetilde{W}_i^{(s)}$ and, on convergence,
appends its rank-$r_s$ truncation $W_i^{(s)}$ to $\Pi_i$. It does not modify any
$W_i^{(j)}$ with $j < s$, by condition (a). Task $t$ evaluates
$W_i^{\mathrm{eff}}(\tau)$, which sums over $j \le \tau$ only. Since
$s > \tau$, the block $W_i^{(s)}$ is absent from that sum. The same holds for
every block appended at any stage between $\tau$ and $s$.

*Encoders.* The variable-length attention layers inside $\phi_m$ receive the
same compress-and-stack treatment as the experts (lines 202–205 of the paper),
so the identical cursor argument applies to each encoder stack.

*Routers.* Stage $s$ instantiates fresh heads $\{G_m^{(s)}\}$. The heads
$\{G_m^{(\tau)}\}$ that task $t$ uses were frozen at the end of stage $\tau$ and
are selected at inference by task identity, not by recency.

*Task head.* $h^{(t)}$ was frozen at the end of stage $\tau$ and is likewise
selected by task identity.

Every operation appearing in the computation graph of $f_t^{(s)}$ is therefore
identical to the corresponding operation in $f_t^{(\tau)}$, on every input.
Pointwise equality of the two functions gives equality of risks by taking
expectations under $\mathcal{D}_t$. $\blacksquare$

**Remark.** The argument is structural: it never refers to the optimizer, the
loss, the task ordering, or the data distribution of any stage. What it does
require is that condition (a) holds for *every* state used by the forward pass,
including buffers such as normalization running statistics that are not
parameters. In our implementation this is what `encoder_freeze_mode =
first_appearance` enforces. Under a freeze mode that leaves such a buffer live,
the claim fails and a small residual drift appears, which is what we observed
before adopting this setting.

### (2) Compression cost, and where it enters the risk

#### (2a) The truncation error is exactly a functional-energy tail

Let $\widetilde{W}_i^{(t)} = \sum_k \sigma_{i,k} u_{i,k} v_{i,k}^\top$ be the
trained stage-$t$ update and $W_i^{(t)}$ its rank-$r_t$ truncation, so the
discarded part is

$$\Delta_i = \widetilde{W}_i^{(t)} - W_i^{(t)} = \sum_{k > r_t} \sigma_{i,k} u_{i,k} v_{i,k}^\top .$$

Let $C_i^{(t)} = \mathbb{E}[z z^\top \mid i \in S_t(x)]$ be the **second
moment** of the inputs actually routed to expert $i$ at stage $t$. (Second
moment, not centred covariance: centring would leave an uncancelled mean term
in the step below.)

**Claim.**

$$\mathbb{E}\big[\|\Delta_i z\|_2^2 \mid i \in S_t(x)\big] = \sum_{k > r_t} \sigma_{i,k}^2\, v_{i,k}^\top C_i^{(t)} v_{i,k} =: \varepsilon_{i,t}(r_t).$$

**Proof.** Using $\|a\|_2^2 = \mathrm{tr}(a a^\top)$ and linearity of trace and
expectation,

$$\mathbb{E}\|\Delta_i z\|_2^2 = \mathbb{E}\,\mathrm{tr}(\Delta_i z z^\top \Delta_i^\top) = \mathrm{tr}(\Delta_i^\top \Delta_i\, C_i^{(t)}).$$

From the singular value decomposition and orthonormality of the left singular
vectors, $u_{i,k}^\top u_{i,l} = \delta_{kl}$, so

$$\Delta_i^\top \Delta_i = \sum_{k > r_t} \sigma_{i,k}^2\, v_{i,k} v_{i,k}^\top .$$

Substituting and using $\mathrm{tr}(v v^\top C) = v^\top C v$ gives the stated
sum. $\blacksquare$

This is an identity, not a bound. The summand
$\sigma_{i,k}^2 v_{i,k}^\top C_i v_{i,k}$ is exactly the per-rank functional
energy $\mathcal{E}_{i,k}$ of Eq. (2), so $\varepsilon_{i,t}(r_t)$ is the tail
of the same quantity plotted in Figs. 3 and 11–15 and in the stage-wise figures
above. The cumulative-energy curves therefore estimate the truncation error
directly rather than through a proxy.

#### (2b) From truncation error to risk

**Conditions.** (i) The loss $\mathcal{L}_t$ is $L_t$-Lipschitz in its first
argument. (ii) Conditional on the trained routers and all other parameters, the
portion of the task-$t$ network downstream of the compressed MoE output is
$\Gamma_t$-Lipschitz. (iii) The active gate weights are nonnegative and sum to
one.

Write $f_{t,\mathrm{full}}$ for task $t$'s predictor at its own stage *before*
truncation and $f_t^{(t)}$ for the stored predictor after it.

**Claim.**

$$R_t(f_t^{(t)}) - R_t(f_{t,\mathrm{full}}) \le L_t \Gamma_t \Big[\sum_{i=1}^{N} \rho_{i,t}\, \varepsilon_{i,t}(r_t)\Big]^{1/2}, \qquad \rho_{i,t} = \mathbb{P}_{x \sim \mathcal{D}_t}(i \in S_t(x)).$$

**Proof.** Fix a stage-$t$ sample $x$ with support $S_t(x)$ and normalized gate
weights $G_i(x)$. Let $e_i(x) = \Delta_i z_i(x)$ be the per-expert perturbation
and

$$\Delta_t(x) = \sum_{i \in S_t(x)} G_i(x)\, e_i(x)$$

the resulting perturbation at the MoE output. Note that $S_t(x)$ is computed
from the router summary $\bar{z}_m$, which does not depend on the expert
matrices, so the support is the same in the full and compressed models and the
two outputs differ only through $\Delta_t(x)$.

By condition (iii) the gate weights form a convex combination, so Jensen's
inequality applied to the convex map $u \mapsto \|u\|_2^2$ gives

$$\|\Delta_t(x)\|_2^2 \le \sum_{i \in S_t(x)} G_i(x)\, \|e_i(x)\|_2^2 \le \sum_{i=1}^{N} \mathbf{1}\{i \in S_t(x)\}\, \|e_i(x)\|_2^2 ,$$

the second step using $G_i(x) \le 1$. Taking expectations and conditioning on
the routing event,

$$\mathbb{E}\|\Delta_t(x)\|_2^2 \le \sum_{i=1}^{N} \rho_{i,t}\, \mathbb{E}\big[\|e_i(x)\|_2^2 \mid i \in S_t(x)\big] = \sum_{i=1}^{N} \rho_{i,t}\, \varepsilon_{i,t}(r_t),$$

by the identity of (2a). Conditions (i) and (ii) give, pointwise,

$$\big|\mathcal{L}_t(f_t^{(t)}(x), y) - \mathcal{L}_t(f_{t,\mathrm{full}}(x), y)\big| \le L_t \Gamma_t \|\Delta_t(x)\|_2 .$$

Taking expectations and applying Cauchy–Schwarz,
$\mathbb{E}\|\Delta_t(x)\|_2 \le (\mathbb{E}\|\Delta_t(x)\|_2^2)^{1/2}$, yields
the claim. $\blacksquare$

**Scope.** The proof uses that truncating the *expert* matrices leaves the
routing support unchanged. It does not extend as stated to compressed encoder
matrices: perturbing an encoder changes $\bar{z}_m$, which is the router's
input, and top-$K$ selection is discontinuous in that input, so $S_t(x)$ may
differ between the full and compressed models. Extending the bound would
require a margin condition on the gate scores together with a term for the
probability of support change. We therefore state the bound for expert
compression only.

#### (2c) The cost is paid once

By (1), $R_t(f_t^{(s)}) = R_t(f_t^{(t)})$ for every $s \ge t$. The compression
error bounded in (2b) is therefore incurred at task $t$'s own stage and does not
grow as further stages arrive. What stream length does consume is capacity:
reserved components accumulate additively along the rank dimension, so
$\sum_{j \le T} r_j \le d$, and a constant per-stage budget $r$ admits at most
$\lfloor d/r \rfloor$ stages. At $d = 128$ and $r = 32$ this gives four, which
matches the stream lengths of Setups 1–4. The bound in (2b) quantifies what is
given up by reducing $r$ to extend the stream.

#### (2d) The stream-level bound

Let $\mathcal{F}_t$ be the complete post-compression predictor class at stage
$t$: the stage-$t$ compressed expert and encoder components, the routers
$\{G_m^{(t)}\}$, and the head $h^{(t)}$, with all earlier components held fixed.
Assume $0 \le \mathcal{L}_t \le 1$, and that the stage-$t$ sample is drawn
independently of the data that determined the components of index $j < t$.

The standard Rademacher uniform-deviation bound gives, for a single stage with
probability at least $1 - \delta_t$, simultaneously for all
$f \in \mathcal{F}_t$,

$$R_t(f) \le \widehat{R}_t(f) + 2\,\mathfrak{R}_{n_t}(\mathcal{L}_t \circ \mathcal{F}_t) + \sqrt{\frac{\log(1/\delta_t)}{2 n_t}} .$$

Setting $\delta_t = \delta/T$ and taking a union bound over $t = 1, \dots, T$
gives the version quoted in our response, holding simultaneously for all stages
with probability at least $1 - \delta$. Applying it to the returned predictor
$f_t^{(t)}$ and averaging over $t$, then using
$R_t(f_t^{(T)}) = R_t(f_t^{(t)})$ from (1), bounds the mean final-stage risk
with no separate forgetting term.

Two caveats we state explicitly. First, $\mathcal{F}_t$ retains the complete
learned class; its complexity cannot be reduced to a union over the
$\binom{N}{K}$ possible supports, because the support is an input-dependent
learned function and the gate values are continuous. Second, we do not estimate
$\mathfrak{R}_{n_t}$ and make no claim that it is tight. Its role here is
structural: it identifies where compression enters the risk and shows that
nothing accumulates across stages.

### (3) Transfer and interference through shared experts

This concerns interaction during shared-expert training, not the
cursor-isolated inference of an already stored task. Let
$E = (E_1, \dots, E_N)$ be the shared expert parameters, hold the routers and
encoders fixed at the point where gradients are evaluated, and write
$g_k = \nabla_E R_k(E)$ and $g_{k'} = \nabla_E R_{k'}(E)$ for two tasks sharing
a modality.

**Claim (i): disjoint supports give zero first-order interaction.** If there
exist disjoint $S_k, S_{k'} \subseteq [N]$ containing the routed supports of
$k$ and $k'$ almost surely, then $\langle g_k, g_{k'} \rangle = 0$.

**Proof.** Write the gradient blockwise,
$g_k = (\nabla_{E_1} R_k, \dots, \nabla_{E_N} R_k)$. Under sparse routing, the
output of $R_k$ has no functional dependence on an expert that task $k$ never
activates, so $\nabla_{E_i} R_k = 0$ for $i \notin S_k$, and likewise
$\nabla_{E_i} R_{k'} = 0$ for $i \notin S_{k'}$. Since
$S_k \cap S_{k'} = \varnothing$, for every index $i$ at least one of the two
blocks vanishes, hence

$$\langle g_k, g_{k'} \rangle = \sum_{i=1}^{N} \langle \nabla_{E_i} R_k, \nabla_{E_i} R_{k'} \rangle = 0 .$$

Equivalently
$\frac{d}{d\eta} R_k(E - \eta g_{k'})\big|_{\eta=0} = -\langle g_k, g_{k'} \rangle = 0$,
so an infinitesimal step on task $k'$ has no first-order effect on $R_k$
through the expert parameters. $\blacksquare$

**Claim (i'): partial overlap.** Exact almost-sure disjointness rarely holds in
practice. If $\rho = \mathbb{P}(S_k(x) \cap S_{k'}(x') \ne \varnothing)$ denotes
the routing overlap probability, then
$|\langle g_k, g_{k'} \rangle| \le \rho\, \|g_k\|\, \|g_{k'}\|$: the inner
product is supported only on the overlapping blocks, and Cauchy–Schwarz on that
restriction gives the stated bound, recovering claim (i) at $\rho = 0$.

**Claim (ii): where supports overlap, the sign decides.** Suppose $R_k$ has a
$\beta_k$-Lipschitz gradient in the expert parameters on the segment between
$E$ and $E^+ = E - \eta g_{k'}$. Then

$$R_k(E^+) \le R_k(E) - \eta \langle g_k, g_{k'} \rangle + \frac{\beta_k \eta^2}{2} \|g_{k'}\|_2^2 .$$

**Proof.** $\beta_k$-smoothness gives, for any increment $\Delta$,

$$R_k(E + \Delta) \le R_k(E) + \langle \nabla_E R_k(E), \Delta \rangle + \frac{\beta_k}{2}\|\Delta\|_2^2 .$$

Substituting $\Delta = -\eta g_{k'}$ gives the claim. If
$\langle g_k, g_{k'} \rangle > 0$, the right-hand side is strictly below
$R_k(E)$ for every step size
$0 < \eta < 2\langle g_k, g_{k'} \rangle / (\beta_k \|g_{k'}\|_2^2)$, so the
update on $k'$ improves task $k$. If the inner product is negative, the
directional derivative of $R_k$ along the update direction is positive and the
step is first-order interfering. $\blacksquare$

**What this does and does not say.** Routing overlap determines whether a
pathway for direct expert-level interaction exists; it does not determine the
sign of that interaction, which is set locally by the gradient inner product on
the shared blocks. The cosine similarity between average routing fingerprints,
which $\mathcal{L}^{(k)}_{\mathrm{div}}$ already regulates and which Fig. 6
reports, is therefore an indicator of sharing *opportunity* rather than a
guarantee of positive transfer. The argument also covers direct interaction
through the shared experts only: disjoint expert support does not preclude
interaction through auxiliary balancing losses or jointly updated statistics.

## C. Instructions to Run:
Under your virtual environment, run
```
pip install -r requirements.txt
```
To train the models, go to dir `clinical-highmmt/src/` and run
```
./run.sh
```
The results will be saved under `clinical-highmmt/src/results/`.

To analyze the weights, go to `clinical-highmmt` and run 
```
./src/analysis/run_analysis.sh
```
Results will be saved under `clinical-highmmt/src/analysis/results/`.
