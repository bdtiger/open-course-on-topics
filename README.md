# Discrete State-Space Models and Expectation-Maximization for Cell-State Trajectories

Project repository for the course *Open Course on Topics in Computer Science*
(1DL710, Period 1, 10 credits, and 1DL705, Period 2, 5 credits) at Uppsala
University.

- **Student:** Arnab Kumar Ghosh, Master's in Computer Science
- **Supervisor:** Prashant Singh
- **Programme coordinator:** Lars-Henrik Eriksson

## Background and aim

When a muscle heals after an ischemic injury (an injury caused by a lack of
blood supply), some immune cells called macrophages change their behaviour.
They switch off their usual immune genes and switch on genes that are typical
of mural cells, the cells that wrap around blood vessels. In other words, they
slowly turn into mural-like cells (Amoedo-Leite et al., 2024).

The data for this process come from single-cell RNA sequencing (scRNA-seq).
Gene activity was measured in single cells at day 0, 7, 14 and 21 after the
injury. The catch is that sequencing destroys the cell, so no cell is ever
measured twice. We only get snapshots, never the history of one cell. The
history has to be inferred.

The aim of this project is to model this change with a **Hidden Markov Model
(HMM)** that has three hidden states:

- **Macrophage**
- **Transitional**
- **Mural-like**

The hidden states cannot be seen directly. What we can see is the gene
activity of each cell, and the model describes it with **Gaussian emissions**.
From the data the model learns three things:

- the initial distribution π (where cells start),
- the transition matrix A (how likely a cell is to move from one state to
  another), and
- the emission parameters (what gene activity looks like in each state).

Learning is done with the **EM algorithm** in its HMM form, called
**Baum-Welch**. The learned transition matrix is then interpreted to say
something about the healing process.

The algorithms (forward-backward and Baum-Welch) are implemented from scratch,
with numerical safeguards such as scaling so that long sequences do not
underflow. The from-scratch code is cross-checked against a library baseline
(`hmmlearn`'s GaussianHMM).

## Relation to ongoing research

The project is part of a collaboration between the supervisor's group
(Division of Scientific Computing) and the group of Prof. Mia Phillipson
(Department of Medical Cell Biology, Uppsala University and SciLifeLab). The
data come from the Phillipson lab and are the basis of Amoedo-Leite et al.
(2024). The project continues an earlier HMM study in the group by Titouan
Ousset (2025-26), which found two open problems that this project takes on:

1. **How to define an observation sequence** from snapshot data. Cells are not
   naturally ordered one after another, so simply lining them up by time point
   mostly produces artefacts of the construction.
2. **Degenerate solutions that depend on initialization.** The earlier model
   produced near-absorbing transition matrices and sudden 0% to 100% jumps.

## Plan of work

1. Study HMM theory from Rabiner (1989) and the EM algorithm from Bilmes (1998).
2. Implement and verify forward-backward and Baum-Welch on a toy sequence with
   known parameters. **Done.**
3. Apply the model to the real data, first with the `hmmlearn` baseline and
   then with the from-scratch pipeline. **In progress.**
4. Fine-tune (initialization, convergence criteria, regularisation of the
   covariances), compare model variants (number of states, feature sets) and
   validate on held-out data.

## What is in this repository

| Folder | What it contains |
|---|---|
| `project-code/` | The Python code. `hmm.py` has the HMM, written from scratch with NumPy. `plot_baum_welch.py` makes the figure used in the progress report. |
| `progress-report/` | The two-week progress reports sent to the supervisor, also in LaTeX. |
| `planning-doc/` | The step-by-step plan that was followed to build the toy HMM. |
| `docs/` | Notes that describe the dataset (what files there are, what is inside them). |
| `requirements.txt` | Python packages needed to run the code. |

The dataset itself is **not** in this repository. It is unpublished data from
the biology group and is too large for git. The notes in `docs/` describe it.

## The code

`project-code/hmm.py` contains, in this order:

- A small toy model with 3 hidden states and 4 observation symbols.
- A sampler that generates fake sequences from the model.
- The **forward** and **backward** algorithms, with scaling so long sequences
  do not underflow.
- The **posteriors** gamma and xi (the E-step).
- **Baum-Welch** re-estimation (the M-step), repeated until the likelihood stops
  improving.

Running the file directly performs a set of sanity checks, for example that the
forward and backward algorithms give the same likelihood and that all
probability rows sum to one.

## How to run

```bash
# 1. Create a virtual environment and install the packages
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the toy HMM with its self-checks
cd project-code
python hmm.py

# 3. Re-create the Baum-Welch figure for the progress report
python plot_baum_welch.py
```

The figure is written to `progress-report/figures/baum_welch.pdf`.

## Building the progress reports

The reports are built with `latexmk` from inside the `progress-report/` folder:

```bash
cd progress-report
latexmk -pdf progress_report_week_3_4.tex
```

The built PDFs are also committed, so you do not need LaTeX to read them.

## Main references

- Amoedo-Leite, C. et al. (2024). The study behind the dataset, on
  macrophage-to-mural-cell reprogramming during tissue repair.
- Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition.* Proceedings of the IEEE.
- Bilmes, J. A. (1998). *A Gentle Tutorial of the EM Algorithm and its
  Application to Parameter Estimation for Gaussian Mixture and Hidden Markov
  Models.*
- Ousset, T. (2026). The earlier HMM study in the group that this project
  continues.
