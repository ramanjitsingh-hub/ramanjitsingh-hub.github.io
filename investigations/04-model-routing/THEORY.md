# Theory for Investigation 04

## Setup
A prompt X is drawn from a distribution D. There are K models. Model k returns an answer with score Y_k in [0, 1] and cost c_k >= 0 (USD, from token counts and list prices). Write mu_k(x) = E[Y_k | X = x] and kappa_k(x) = E[c_k | X = x]. A router maps a prompt to a distribution pi(x) over models, so randomised routers are allowed. Its value is V(pi) = E[sum_k pi_k(X) Y_k] and its cost Q(pi) = E[sum_k pi_k(X) c_k].

The budgeted problem is: maximise V(pi) subject to Q(pi) <= B.

## Proposition 1 (the Lagrangian rule and the shape of the frontier)
For lambda >= 0 let pi_lambda(x) = argmax_k mu_k(x) - lambda kappa_k(x).
1. pi_lambda maximises V - lambda Q over all routers.
2. If Q(pi_lambda) = B, pi_lambda solves the budgeted problem.
3. The frontier F(B) = max{V(pi) : Q(pi) <= B} is concave and nondecreasing. Its graph is the upper concave envelope of the points (Q(pi_lambda), V(pi_lambda)); at a kink, mixing the two adjacent lambda-routers attains every point on the segment.

Proof. V - lambda Q = E[sum_k pi_k(X)(mu_k(X) - lambda kappa_k(X))] by the tower rule, and the integrand is maximised pointwise by putting all mass on the argmax, which gives 1. For 2, if Q(pi) <= B = Q(pi_lambda) then V(pi) <= V(pi_lambda) - lambda (Q(pi_lambda) - Q(pi)) <= V(pi_lambda). For 3, the set of achievable (Q, V) pairs is convex because routers can be mixed and V, Q are linear in pi, so its upper boundary is concave; every supporting line has slope lambda >= 0 and touches the set at pi_lambda. QED.

Consequence for evaluation. A router that knows nothing about the prompt has mu_hat and kappa_hat constant in x, so it can only pick a single model, and by mixing it reaches exactly the upper hull of the single models' (cost, score) points. That hull, not any one model, is the right baseline. A claim such as "matches model A at a fraction of its cost" is only evidence of routing skill if it beats a coin-flip mix of two fixed models.

## Proposition 2 (plug-in regret)
Let pi_hat route by argmax_k mu_hat_k - lambda kappa_hat_k. Then
(V - lambda Q)(pi_lambda) - (V - lambda Q)(pi_hat) <= 2 E[max_k |mu_hat_k(X) - mu_k(X)| + lambda |kappa_hat_k(X) - kappa_k(X)|].

Proof. Write u = mu - lambda kappa and u_hat likewise. With k* = pi_lambda(x) and k_hat = pi_hat(x),
u(k*) - u(k_hat) = [u(k*) - u_hat(k*)] + [u_hat(k*) - u_hat(k_hat)] + [u_hat(k_hat) - u(k_hat)].
The middle term is <= 0 by the definition of k_hat, and each outer term is at most max_k |u_k - u_hat_k|. Take expectations. QED.

The bound is loose where one model clearly dominates, because there an error in mu_hat does not change the decision. It cannot be checked directly on data, since mu is never observed, only one draw of Y per model and prompt.

## Proposition 3 (the hindsight oracle overstates what is reachable)
For every router pi, V(pi) <= E[max_k mu_k(X)] <= E[max_k Y_k].

Proof. V(pi) = E[sum_k pi_k mu_k] <= E[max_k mu_k]. The second inequality is Jensen's, since max is convex: max_k mu_k(x) = max_k E[Y_k | x] <= E[max_k Y_k | x]. QED.

The gap E[max Y] - E[max mu] is luck: the oracle is credited whenever any of K noisy answers happens to be right. With K = 13 this gap can be large even when no prompt feature predicts which model will be right.

## Certification (Section 6 of the page)
Order the lambdas from expensive to cheap, lambda_1 < lambda_2 < .... For each, test H_j: V(pi_lambda_j) < target on n calibration prompts and walk forward while H_j is rejected at level delta, stopping at the first failure (fixed-sequence testing; Angelopoulos, Bates, Candes, Jordan and Lei, "Learn then Test", 2021). Fixed-sequence testing controls the family-wise error at delta without any multiplicity correction, so P(V(pi_chosen) < target) <= delta.

Two p-values are used. Hoeffding: p = exp(-2 n (mean - target)^2) when mean > target. Betting: the capital process K_t = prod_{i<=t} (1 + eta_i (x_i - target)) with predictable eta_i in [0, 0.5/target] is a nonnegative supermartingale under H_j, so by Ville's inequality P(sup_t K_t >= 1/delta) <= delta and 1/sup_t K_t is a valid p-value (Waudby-Smith and Ramdas, 2024). The betting test adapts to the variance and is much less conservative when scores are concentrated.
