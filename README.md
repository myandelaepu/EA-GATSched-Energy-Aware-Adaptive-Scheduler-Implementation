# Rerun: residual-over-SJBF backfill policy (GAT-BF-R)

## What is different from v2
GAT-BF-R scores each backfill candidate as `logit = -1.0 * (SJBF rank) + learned residual`. The residual network is
initialised at zero, so **the starting policy makes exactly SJBF's decisions** (checked: validation reward vs SJBF is
exactly 0 at episode 0, and test-window metrics equal SJBF's). REINFORCE then fine-tunes the residual against SJBF.
A fine-tuned checkpoint replaces SJBF only if it beats SJBF on the validation windows by a reward margin of 0.01;
otherwise the reported policy is SJBF. The candidates are the 32 shortest requested walltimes, so SJBF's choice is
always available. Other changes: batch of 8 episodes, 240 episodes, learning rate 3e-4, entropy 0.003, and half of the
training episodes are drawn from congested training windows (probability proportional to SJBF wait, computed on
training windows only). Reward is 0.6 slowdown + 0.4 wait relative to SJBF, as for GAT-BF-WS.
Windows, splits, simulator, metrics, seeds (0, 1, 2) and the 85% cap are unchanged. Baselines FCFS, EASY, SJBF and Random-BF
are rerun and must reproduce the v2 numbers exactly (a check on your setup).

## Rules for using the result (so that it holds up)
1. Run it once with the constants as written. They are fixed in advance. Do not change them after seeing test results.
2. Whatever the outcome, it is reported. If you try other settings, say how many you tried and report all of them.
3. The constants were chosen without looking at test windows. Validation windows are used only for the checkpoint decision.

## Run (Colab, same as v2)
Upload the four `ANL-ALCF-DJC-*.csv.gz` files to `/content/data`, then run `ea_gatsched_v3_residual.py`.
Smoke test first: `EA_FAST=1`. Full run: expect about 2.5 to 3 hours on a CPU runtime (18 models, 240 episodes each).
Save `ea_gatsched_results.zip` to Drive straight away.
Send back: `per_window_results.csv`, `summary.csv`, `paired_vs_easy.csv`, `run_log.txt`, `config.json`.

## How to read the outcome
* If GAT-BF-R stays SJBF in a cell (log says "policy stays SJBF"), it ties SJBF there by construction.
* A genuine improvement shows as GAT-BF-R below SJBF with an interval that excludes zero after Holm correction.
  Small tie-or-better results are the most likely outcome; a clear win is possible but not expected.
