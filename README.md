# Take-Home Assignment: Syllable Discovery in Sequential Data

As ML engineers at Synature, a lot of what we do comes down to modeling animal voices as structured signals in some representation space. This assignment is built in that spirit: it's a synthetic version of syllable-structured, sequential data we work with day to day.

**Time budget: ~5 hours.** When you hit 5 hours, stop, and note in your writeup what you would do next.

![Deadline](https://img.shields.io/badge/Deadline-23.08.2026-red) Please send your solution to gabriel@synature.ch by **23.08.2026, 23:59 CEST (UTC+2)**. We're more interested in how you approached the problem than in polished code.

## Background

`markov_circles_timeseries.py` generates a synthetic "song": a 20-dimensional time series that switches between 10 latent states ("syllables").

In this task the data is generated with `--subspace-dim 4`: all 10 circle planes are packed into a shared 4-dimensional subspace.

> **Note:** `data/data.npz` and `data/config.json` in this repo were already generated with `--subspace-dim 4` — you don't need to regenerate anything to get started. `markov_circles_timeseries.py` is the actual generating script, so feel free to inspect it to understand the data-generating process, and feel free to re-run it with different flags if you want to play around and build intuition. Just make sure your final submitted results are evaluated against the provided `data/data.npz`.

## The task

1. Use the provided dataset:

   ```bash
   ls data/data.npz data/config.json
   ```

   This is the overlapping (`--subspace-dim 4`) dataset your solution should be evaluated on. If you want to regenerate it yourself or experiment with other settings, you can re-run:

   ```bash
   python markov_circles_timeseries.py --subspace-dim 4 --no-umap
   ```

2. Build a method that assigns a syllable label to every one of the 100,000 time steps, using **only the observations `X` and their time order**.

   Everything else in `data.npz` (`states`, `thetas`, `transition_matrix`, `radii`, `entry_angles`, `periods`) is ground truth. You may load it **only** in your final evaluation script.

   You may assume there are 10 syllables.

3. Evaluate your labels against the ground truth. Labels are unordered, so use permutation-invariant metrics: report **ARI** and **NMI**, plus a confusion matrix after optimal (Hungarian) matching, and a plot comparing predicted vs. true label sequences over a few thousand-step windows.

## Deliverables

- Working code (repo link) that reproduces your result end to end
- A short writeup (~2 page): your approach and why it fits this data, what you tried that didn't work, your metrics, and known failure modes

## Ground rules

- Any language and libraries; Python with numpy/scipy/scikit-learn is plenty. No GPU is needed.
- **AI assistants (ChatGPT, Claude, Copilot, Cursor, etc.) are allowed and encouraged.** The requirement is that you can explain and defend every part of the solution — the method, the code, and the results — in a follow-up conversation. Letting an LLM write code is fine; submitting a method you can't explain is disqualifying.

## What we look at

Roughly in order of weight: (1) whether the method is well-matched to the structure of this data, (2) honest, correct evaluation, (3) clarity of the writeup and the follow-up discussion, (4) code quality and reproducibility.

## Requirements

```
numpy
matplotlib
umap-learn
scikit-learn
scipy
```
