Entanglement Index Calculator
A Python tool for detecting and quantifying acausal resonances (stubborn co-occurrences) in pairs or groups of events.

This script analyzes time-series or sequential data to find patterns that go beyond simple correlation, causation, or coincidence — what the author calls "entanglement" or acausal resonance.

It supports:

2 events (classic mode)

3 events (pairwise + 3-way interaction information)

Full suite of metrics: original PMI, smoothed, normalized, weighted, resonant (adaptive decay), symmetric, synergy, kernelized (non-linear), discord proxy, multi-scale wavelet spectrum

Jupyter notebook interface with dropdowns for conditions and wavelet family

Automatic plotting of multi-scale spectrum per pair

Features

Flexible event binarization — numerical → above_median/above_mean, binary → None, custom thresholds
Adaptive temporal decay — learns optimal λ to weight recent data
Multi-scale analysis — wavelet decomposition (Haar, Daubechies, etc.) to see short-term vs long-term patterns
Multi-event support — computes all pairwise indices + 3-way interaction I(A;B;C)
Non-linear detection — RBF kernel on raw values
Quantum-inspired discord proxy — highlights "extra" dependence beyond linear correlation
Bootstrap CI on original index (optional)
Interactive Jupyter UI — change path, conditions, wavelet family easily

Requirements

pip install pandas numpy scipy matplotlib pywavelets scikit-learn ipywidgets

pandas — Excel reading and data handling
numpy, scipy — numerical computations
matplotlib — plotting multi-scale spectrum
pywavelets — wavelet decomposition
scikit-learn — RBF kernel for non-linear detection
ipywidgets — interactive dropdowns in Jupyter

Environment

Best run in Jupyter Notebook or JupyterLab (Google Colab, VS Code Jupyter, etc.).

Installation

Save the script as entanglement_index.py or paste into a Jupyter notebook cell.
Install dependencies (see above).
Prepare your Excel file (see Data Format below).

Column name	Description	Example values	Required?
Time	Optional date/time for sorting	2024-01-01, 2025-12-31	No
Event A	First event (numerical or binary)	BTC close price, 0/1 flags	Yes
Event B	Second event (numerical or binary)	ETH close price, volume	Yes
Event C	Optional third event	XRP close price, moon phase 0/1	No
<img width="785" height="121" alt="image" src="https://github.com/user-attachments/assets/7f4696ae-b8ff-41f2-8a30-d5de1fa38542" />
Numerical columns → binarized with conditions (above_median, above_mean, positive, etc.)
Binary/0-1 columns → use None as condition
Multiple events → add "Event D", "Event E", etc. (pairwise will include them)

How to Use
Step 1: Open in JupyterPaste the entire script into a Jupyter notebook cell and run it.
Step 2: Use the Interface
You will see:
Excel Path text box (defaults to your BTC file)
A Condition, B Condition, C Condition dropdowns
Wavelet Family dropdown (haar, db4, sym4, coif1)
Load Excel & Compute button

Click the button → results appear below, including:All pairwise metrics (Original E, Resonant E, Symmetric, Synergy, Kernelized, Discord, Multi-Scale Avg)
3-way interaction I(A;B;C) if Event C present
Multi-scale spectrum plot for each pair

Step 3: Interpret Results
See the separate guide: How to Interpret Entanglement Index (interpretation-guide.md)

Quick cheat sheet:

Metric	Positive means	Negative means	Typical range	Key insight
Original E	A boosts B	A suppresses B	-5 to +5	Base linkage
Resonant E	Recent strong link	Recent suppression	-5 to +5	Time shift?
Symmetric	Mutual attraction	Mutual repulsion	-5 to +5	Bidirectional
Synergy Resonance	Extra 3-way boost	Lagged explains away	-5 to +5	Higher-order?
Kernelized	Non-linear attraction	Non-linear suppression	-10 to +10	Complex patterns
Discord Proxy	Non-classical extra	Extra suppression	-5 to +5	Beyond linear
Multi-Scale Avg	Fluctuations co-occur	Fluctuations repel	-3 to +3	Scale-invariant?
I(A;B;C)	3-way synergy	Redundancy (C explains A-B)	-5 to +5	Multi-event key
<img width="872" height="217" alt="image" src="https://github.com/user-attachments/assets/1712b843-f588-41af-a3d7-ed9ebb412571" />

Step 4: Customize

Change Excel path in the text box
Select different conditions (e.g., positive for returns, None for binary flags)
Try different wavelets (db4 often smoother)

Step 5: Save & Share Results

Results print to output. To save:

import json
with open('entanglement_results.json', 'w') as f:
    json.dump(result, f, indent=4)

Data Interpretation Guide

See the detailed interpretation guide (separate markdown file recommended) or the in-script print explanations.

Quick rules of thumb:

Positive + stubborn multi-scale → possible "entanglement" candidate

Negative multi-scale → events avoid each other in dynamics

Negative I(A;B;C) → redundancy (classical explanation likely)

Positive I(A;B;C) → true higher-order synergy (intriguing!)

λ < 0.9 → recent regime shift

Applications

Cryptocurrency regime detection
Synchronicity / coincidence research
Neuroscience signal co-occurrence
Climate extreme event patterns
Personal log analysis (mood, dreams, events)

Contributing

Feel free to fork, add 4+ event support, better plots, or export features. PRs welcome!









