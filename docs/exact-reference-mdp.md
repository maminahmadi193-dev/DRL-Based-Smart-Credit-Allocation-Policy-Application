# Exact finite reference MDP

This is a small algorithmic benchmark for checking learned action-values against an exactly computable \(Q^*\). It is **not** the full portfolio simulator.

## What is represented

- Two fixed, unique applicants and seven reachable decision states.
- The six original actions: 0, 100M, 200M, 300M, 400M and 500M.
- A 500M initial budget, represented in 100M units.
- The current production social-reward calculation and customer-risk update methods.
- Stochastic missed-payment outcomes; a miss delays principal liquidity past the next applicant decision and incurs the configured default penalty.
- Exact Bellman backward induction for \(Q^*\), followed by tabular Q-learning and (optionally) SB3 DQN comparisons.

## Important scope limitation

To make exact enumeration transparent and fast, each toy loan is fully amortized after one period. A missed principal is ultimately collected at settlement; with \(\gamma=1\), this toy retains the undiscounted episode-reward contribution but does **not** reproduce the full 18-month contract lifecycle, 90+ DPD recognition timing, repeat applications, or the production hazard calibration. The hazard ceiling defaults to 0.20 to make stochastic outcomes visible and is an algorithmic stress-test setting, not an empirical estimate of bank default probability.

Consequently, the computed \(Q^*\) is exact **only for this reduced benchmark**. It cannot establish convergence of DQN in the full portfolio environment.

## Run on Windows

From the repository root, using the project virtual environment:

\`\`\`powershell
python env/test_exact_reference_mdp.py
python env/exact_reference_mdp.py --skip-dqn --q-learning-episodes 500000 --seed 42
\`\`\`

The first command runs unit tests. The second computes exact \(Q^*\), trains tabular Q-learning, reports the learned policy's exact expected return and Q-error, and writes:

- \`models/results/exact_reference_mdp/exact_vs_tabular_q.csv\`
- \`models/results/exact_reference_mdp/summary.json\`

To add the neural-network comparison:

\`\`\`powershell
python env/exact_reference_mdp.py --q-learning-episodes 500000 --dqn-timesteps 50000 --seed 42
\`\`\`

The script requires the project's installed \`gymnasium\` and \`stable-baselines3\` packages for the DQN comparison. Q-learning uses exploring starts and a Robbins–Monro-style learning rate \(\alpha_n=1/n^{0.6}\), with persistent exploration. This is designed to support state-action coverage in the finite benchmark; the finite run still reports empirical error rather than claiming that a finite number of episodes proves convergence.
