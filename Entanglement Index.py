# =============================================================================
# Entanglement Index Calculator - Full Multi-Event + All Evolutions Version
# =============================================================================
# Features:
# - Loads Event A, B, optional Event C from Excel
# - Computes ALL pairwise (A→B, A→C, B→C) with full metrics (original, smoothed, normalized, weighted, resonant, symmetric, kernelized, discord)
# - Computes 3-way interaction I(A;B;C) if Event C present
# - Adaptive decay, multi-scale wavelet spectrum + plot
# - Bootstrap CI on original A→B
# - User-adjustable Excel path, conditions for A/B/C, wavelet family
#
# Dependencies: pywavelets, scikit-learn, pandas, numpy, scipy, matplotlib, ipywidgets
# Install if needed: !pip install pywavelets scikit-learn
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pywt
from sklearn.metrics.pairwise import rbf_kernel
from typing import Union, Tuple, Optional, Callable, List, Dict
import ipywidgets as widgets
from IPython.display import display
import os

DEFAULT_EXCEL_PATH = r"C:\Users\oliva\OneDrive\Documents\Excel doc\BTC.xlsx"

def load_events_from_excel(path: str = DEFAULT_EXCEL_PATH) -> Dict[str, pd.Series]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {path}")

    try:
        df = pd.read_excel(path)
    except Exception as e:
        raise ValueError(f"Error reading Excel: {str(e)}")

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors='coerce')
        df = df.sort_values("Time").reset_index(drop=True)
        print("Data sorted by 'Time'.")

    event_cols = [col for col in df.columns if col.startswith("Event ")]
    if len(event_cols) < 2:
        raise ValueError("Excel must have at least 'Event A' and 'Event B'.")

    events = {col: df[col] for col in event_cols}

    print(f"Loaded {len(df)} rows from {path}")
    for name, s in events.items():
        print(f"{name} range: {s.min():.2f} to {s.max():.2f}")

    return events


def define_event_series(
    data: Union[list, np.ndarray, pd.Series],
    event_condition: Union[str, float, Callable, None] = None
) -> pd.Series:
    if isinstance(data, (list, np.ndarray)):
        s = pd.Series(data)
    elif isinstance(data, pd.Series):
        s = data.copy()
    else:
        raise ValueError("Input data must be list, numpy array or pandas Series")
    
    s = s.dropna().reset_index(drop=True)
    
    if event_condition is None:
        if s.dtype == bool:
            return s.astype(int)
        try:
            return (s.astype(bool)).astype(int)
        except:
            raise ValueError("Cannot interpret data as boolean automatically. Provide event_condition.")
    
    elif isinstance(event_condition, (int, float)):
        return (s >= event_condition).astype(int)
    
    elif isinstance(event_condition, str):
        cond = event_condition.lower()
        if cond == 'positive':
            return (s > 0).astype(int)
        elif cond == 'negative':
            return (s < 0).astype(int)
        elif cond == 'nonzero':
            return (s != 0).astype(int)
        elif cond == 'above_mean':
            return (s > s.mean()).astype(int)
        elif cond == 'above_median':
            return (s > s.median()).astype(int)
        else:
            raise ValueError(f"Unknown string condition: {event_condition}")
    
    elif callable(event_condition):
        result = event_condition(s)
        if not np.issubdtype(result.dtype, np.bool_):
            raise ValueError("Custom function must return boolean Series")
        return result.astype(int)
    
    else:
        raise ValueError("event_condition must be None, number, string preset or callable")


def find_optimal_decay_lambda(
    A: np.ndarray,
    B: np.ndarray,
    alphas: List[float] = [0.5, 1.0],
    n_splits: int = 5,
    lambda_grid: np.ndarray = np.linspace(0.8, 1.0, 11),
    base: float = 2
) -> Tuple[float, float]:
    n = len(A)
    if n < 10:
        return 0.95, 0.0

    best_lambda = 0.95
    best_score = -np.inf

    fold_size = max(1, n // (n_splits + 1))
    for lam in lambda_grid:
        scores = []
        for i in range(n_splits):
            train_end = (i + 1) * fold_size
            val_start = train_end
            val_end = min(val_start + fold_size, n)

            if val_end <= val_start:
                continue

            A_train, B_train = A[:train_end], B[:train_end]
            A_val, B_val = A[val_start:val_end], B[val_start:val_end]

            t_train = np.arange(len(A_train))
            weights_train = lam ** (len(A_train) - 1 - t_train)
            w_sum_train = weights_train.sum()
            w_A_train = np.sum(weights_train * A_train)
            w_B_train = np.sum(weights_train * B_train)
            w_AB_train = np.sum(weights_train * (A_train & B_train))

            p_B_train = (w_B_train + alphas[0]) / (w_sum_train + 2 * alphas[0])
            p_B_given_A_train = (w_AB_train + alphas[0]) / (w_A_train + alphas[0] + alphas[1]) if w_A_train > 0 else alphas[0] / w_sum_train

            ll = 0
            n_val = len(A_val)
            for a, b in zip(A_val, B_val):
                if a == 1:
                    prob = p_B_given_A_train if b == 1 else (1 - p_B_given_A_train)
                else:
                    prob = p_B_train if b == 1 else (1 - p_B_train)
                ll += np.log(prob + 1e-10)

            scores.append(ll / n_val if n_val > 0 else 0)

        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_lambda = lam

    return best_lambda, best_score


def compute_multi_scale_rei(
    A: np.ndarray,
    B: np.ndarray,
    scales: List[int] = [1, 2, 4, 8],
    decay_lambda: float = 0.95,
    alpha_joint: float = 0.5,
    alpha_non: float = 1.0,
    base: float = 2,
    wavelet_family: str = 'haar'
) -> Dict[str, Union[Dict[str, float], float]]:
    spectrum = {}
    avg_rei = 0.0
    n_scales = 0

    for scale in scales:
        if scale > len(A) // 2:
            continue

        try:
            coeffs_A = pywt.wavedec(A, wavelet_family, level=scale)
            coeffs_B = pywt.wavedec(B, wavelet_family, level=scale)
            detail_A = coeffs_A[-1][:min(len(coeffs_A[-1]), len(coeffs_B[-1]))] if len(coeffs_A) > 1 else A
            detail_B = coeffs_B[-1][:len(detail_A)] if len(coeffs_B) > 1 else B
        except Exception as e:
            print(f"Wavelet decomposition failed at scale {scale}: {e}")
            continue

        if len(detail_A) < 5:
            continue

        detail_A_bin = (detail_A > 0).astype(int)
        detail_B_bin = (detail_B > 0).astype(int)

        t = np.arange(len(detail_A_bin))
        weights = decay_lambda ** (len(detail_A_bin) - 1 - t)
        w_sum = weights.sum()
        w_A = np.sum(weights * detail_A_bin)
        w_B = np.sum(weights * detail_B_bin)
        w_AB = np.sum(weights * (detail_A_bin & detail_B_bin))

        p_B_rei = (w_B + alpha_joint) / (w_sum + 2 * alpha_joint)
        p_B_given_A_rei = (w_AB + alpha_joint) / (w_A + alpha_joint + alpha_non) if w_A > 0 else alpha_joint / (w_sum + 2 * alpha_joint)

        rei_scale = 0.0
        if p_B_given_A_rei > 0 and p_B_rei > 0:
            rei_scale = np.log(p_B_given_A_rei / p_B_rei) / np.log(base)

        spectrum[f"scale_{scale}"] = rei_scale
        avg_rei += rei_scale
        n_scales += 1

    if n_scales > 0:
        avg_rei /= n_scales

    return {
        'MultiScale_REI_Spectrum': spectrum,
        'MultiScale_Avg_REI': avg_rei
    }


def plot_multi_scale_spectrum(result: dict, pair_name: str = "A→B"):
    if 'MultiScale_REI_Spectrum' in result and result['MultiScale_REI_Spectrum']:
        scales = list(result['MultiScale_REI_Spectrum'].keys())
        values = list(result['MultiScale_REI_Spectrum'].values())
        plt.figure(figsize=(10, 5))
        plt.bar(scales, values, color='teal', alpha=0.7)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.title(f"Multi-Scale Entanglement Spectrum ({pair_name})")
        plt.xlabel("Wavelet Scale")
        plt.ylabel("REI Value")
        plt.ylim(-1.2, 1.2)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    else:
        print(f"No multi-scale data to plot for {pair_name}.")


# ──────────────────────────────────────────────────────────────────────────────
#                     MAIN MULTI-EVENT FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def entanglement_index_multi(
    excel_path: str = DEFAULT_EXCEL_PATH,
    conditions: Dict[str, Union[str, float, Callable, None]] = None,
    base: float = 2,
    bootstrap_ci: bool = True,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    random_state: int = 42,
    smoothing_alpha: float = 1.0,
    weight_type: str = 'sqrt_nA',
    decay_lambda: float = 0.95,
    alpha_joint: float = 0.5,
    alpha_non: float = 1.0,
    adaptive_decay: bool = True,
    wavelet_scales: List[int] = [1, 2, 4, 8],
    wavelet_family: str = 'haar'
) -> Dict:
    """
    Main multi-event function: loads events, computes pairwise + 3-way interaction.
    """
    if conditions is None:
        conditions = {
            'Event A': 'above_median',
            'Event B': 'above_median',
            'Event C': 'None'  # default for optional C
        }

    events_raw = load_events_from_excel(excel_path)

    # Binarize all present events
    binary_events = {}
    for name in events_raw:
        cond = conditions.get(name, 'above_median')
        binary_events[name] = define_event_series(events_raw[name], cond)

    # Align lengths
    min_len = min(len(s) for s in binary_events.values())
    for name in binary_events:
        binary_events[name] = binary_events[name][:min_len]

    n = min_len
    result = {'pairwise': {}, 'multipartite': {}}

    # ─── Compute all pairwise ───────────────────────────────────────────────────
    event_names = list(binary_events.keys())
    for i in range(len(event_names)):
        for j in range(i+1, len(event_names)):
            name1, name2 = event_names[i], event_names[j]
            A = binary_events[name1]
            B = binary_events[name2]
            raw_A = events_raw[name1].values[:n]
            raw_B = events_raw[name2].values[:n]

            pair_key = f"E({name1} → {name2})"
            pair_result = {}

            # Original
            p_B = B.mean()
            if p_B == 0:
                pair_result['error'] = "P(B) = 0"
            else:
                p_B_given_A = B[A == 1].mean() if A.sum() > 0 else 0
                ei_original = -np.inf if p_B_given_A == 0 else np.log(p_B_given_A / p_B) / np.log(base)
                pair_result.update({
                    'Entanglement_Index_Original': ei_original,
                    'P(B)': p_B,
                    'P(B|A)': p_B_given_A,
                    'P(A)': A.mean(),
                    'n_samples': n,
                    'n_A_occurrences': A.sum(),
                    'n_B_occurrences': B.sum(),
                    'n_joint_occurrences': (A & B).sum()
                })

                # Smoothed
                n_AB = (A & B).sum()
                p_B_sm = (B.sum() + smoothing_alpha) / (n + 2 * smoothing_alpha)
                p_B_given_A_sm = (n_AB + smoothing_alpha) / (A.sum() + 2 * smoothing_alpha) if A.sum() > 0 else smoothing_alpha / (n + 2 * smoothing_alpha)
                ei_smoothed = 0.0 if p_B_given_A_sm <= 0 or p_B_sm <= 0 else np.log(p_B_given_A_sm / p_B_sm) / np.log(base)
                pair_result['Entanglement_Index_Smoothed'] = ei_smoothed

                # Normalized
                p_AB = n_AB / n
                npmi = 0.0
                if p_AB > 0 and np.isfinite(ei_original):
                    npmi = ei_original / (-np.log(p_AB) / np.log(base))
                pair_result['Entanglement_Index_Normalized'] = npmi

                # Weighted
                weight = np.sqrt(A.sum() / n) if n > 0 else 0.0
                ei_weighted = ei_original * weight if np.isfinite(ei_original) else 0.0
                pair_result['Entanglement_Index_Weighted'] = ei_weighted

                # Resonant + Adaptive
                decay_lambda_used = decay_lambda
                adaptive_score = None
                if adaptive_decay and n > 20:
                    try:
                        opt_lambda, opt_score = find_optimal_decay_lambda(
                            A.values if isinstance(A, pd.Series) else A,
                            B.values if isinstance(B, pd.Series) else B,
                            alphas=[alpha_joint, alpha_non],
                            base=base
                        )
                        decay_lambda_used = opt_lambda
                        adaptive_score = opt_score
                    except Exception as e:
                        print(f"Adaptive failed for {pair_key}: {e}")

                t = np.arange(n)
                weights = decay_lambda_used ** (n - 1 - t)
                w_sum = weights.sum()
                w_A = np.sum(weights * A)
                w_B = np.sum(weights * B)
                w_AB = np.sum(weights * (A & B))

                p_B_rei = (w_B + alpha_joint) / (w_sum + 2 * alpha_joint)
                p_B_given_A_rei = (w_AB + alpha_joint) / (w_A + alpha_joint + alpha_non) if w_A > 0 else alpha_joint / (w_sum + 2 * alpha_joint)

                rei = 0.0 if p_B_given_A_rei <= 0 or p_B_rei <= 0 else np.log(p_B_given_A_rei / p_B_rei) / np.log(base)

                pair_result.update({
                    'Resonant_Entanglement_Index': rei,
                    'decay_lambda_used': decay_lambda_used,
                    'adaptive_decay': adaptive_decay,
                    'adaptive_lambda_score': adaptive_score
                })

                # Symmetric + Synergy
                p_A = A.mean()
                p_A_given_B = A[B == 1].mean() if B.sum() > 0 else 0
                ei_reverse = -np.inf if p_A_given_B == 0 else np.log(p_A_given_B / p_A) / np.log(base)
                symmetric_e = (rei + (ei_reverse if np.isfinite(ei_reverse) else 0)) / 2

                synergy = symmetric_e
                if n > 1:
                    lagged_A = A.shift(1).fillna(0) if isinstance(A, pd.Series) else np.roll(A, 1)
                    lagged_A[0] = 0
                    p_B_given_laggedA = B[lagged_A == 1].mean() if np.sum(lagged_A == 1) > 0 else p_B
                    lagged_cond = 0.0 if p_B_given_laggedA <= 0 or p_B <= 0 else np.log(p_B_given_laggedA / p_B) / np.log(base)
                    synergy = symmetric_e - lagged_cond

                pair_result['Symmetric_Entanglement_Index'] = symmetric_e
                pair_result['Synergy_Resonance'] = synergy

                # Kernelized
                kernel_rei = 0.0
                if n > 10:
                    K = rbf_kernel(raw_A.reshape(-1, 1), raw_B.reshape(-1, 1), gamma=1.0)
                    p_B_kernel = np.mean(K[A == 1]) if np.sum(A == 1) > 0 else 0
                    kernel_rei = np.log(p_B_kernel / p_B) / np.log(base) if p_B_kernel > 0 and p_B > 0 else 0.0
                pair_result['Kernelized_REI'] = kernel_rei

                # Discord Proxy
                classical_corr = np.corrcoef(raw_A, raw_B)[0,1] if n > 1 else 0.0
                discord_proxy = rei - classical_corr
                pair_result['Discord_Proxy'] = discord_proxy

                # Multi-Scale (only for this pair)
                multi_scale = compute_multi_scale_rei(
                    A.values if isinstance(A, pd.Series) else A,
                    B.values if isinstance(B, pd.Series) else B,
                    scales=wavelet_scales,
                    decay_lambda=decay_lambda_used,
                    alpha_joint=alpha_joint,
                    alpha_non=alpha_non,
                    base=base,
                    wavelet_family=wavelet_family
                )
                pair_result.update(multi_scale)

            result['pairwise'][pair_key] = pair_result

    # ─── Multipartite (3-way) if 3+ events ──────────────────────────────────────
    if len(event_names) >= 3:
        A = binary_events[event_names[0]]
        B = binary_events[event_names[1]]
        C = binary_events[event_names[2]]
        p_AB = (A & B).mean()
        p_AB_given_C = ((A & B) & C).sum() / C.sum() if C.sum() > 0 else 0
        ei_ab = result['pairwise'][f"E({event_names[0]} → {event_names[1]})"]['Entanglement_Index_Original']
        ei_ab_given_c = -np.inf if p_AB_given_C == 0 else np.log(p_AB_given_C / p_AB) / np.log(base) if p_AB > 0 else 0
        interaction = ei_ab - ei_ab_given_c if np.isfinite(ei_ab) and np.isfinite(ei_ab_given_c) else 0.0

        result['multipartite'] = {
            'I(A;B;C)': interaction,
            'interpretation': "Positive = synergy (3-way stronger), Negative = redundancy (C explains A-B link)"
        }

    return result


# ──────────────────────────────────────────────────────────────────────────────
#                     PRINT FUNCTION (multi-pair aware)
# ──────────────────────────────────────────────────────────────────────────────

def print_entanglement_result(result: dict):
    print("═" * 100)
    print("ENTANGLEMENT INDEX RESULTS - Multi-Event Version")
    print("═" * 100)

    def interp(e):
        if e > 1.0: return "Strong positive"
        if e > 0.5: return "Moderate positive"
        if e > 0.0: return "Weak positive"
        if e < -1.0: return "Strong negative (suppression)"
        if e < -0.5: return "Moderate negative"
        return "Very weak / negligible"

    for pair, metrics in result['pairwise'].items():
        print(f"\nPair: {pair}")
        print(f"  Original E          = {metrics.get('Entanglement_Index_Original', 0):.4f}")
        print(f"  Resonant E (λ={metrics.get('decay_lambda_used', 0.95):.3f}) = {metrics.get('Resonant_Entanglement_Index', 0):.4f}")
        print(f"  Symmetric           = {metrics.get('Symmetric_Entanglement_Index', 0):.4f}")
        print(f"  Synergy Resonance   = {metrics.get('Synergy_Resonance', 0):.4f}")
        print(f"  Kernelized          = {metrics.get('Kernelized_REI', 0):.4f}")
        print(f"  Discord Proxy       = {metrics.get('Discord_Proxy', 0):.4f}")
        if 'MultiScale_Avg_REI' in metrics:
            print(f"  Multi-Scale Avg     = {metrics['MultiScale_Avg_REI']:.4f}")
            plot_multi_scale_spectrum(metrics, pair)

    if 'multipartite' in result:
        print("\nMultipartite Interaction:")
        for key, val in result['multipartite'].items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {val}")


# ──────────────────────────────────────────────────────────────────────────────
#                     MAIN INTERFACE
# ──────────────────────────────────────────────────────────────────────────────

path_input = widgets.Text(
    value=DEFAULT_EXCEL_PATH,
    placeholder='Enter full Excel path',
    description='Excel Path:',
    layout=widgets.Layout(width='700px')
)

a_cond = widgets.Dropdown(
    options=['None', 'positive', 'negative', 'nonzero', 'above_mean', 'above_median'],
    value='above_median',
    description='A Condition:'
)

b_cond = widgets.Dropdown(
    options=['None', 'positive', 'negative', 'nonzero', 'above_mean', 'above_median'],
    value='above_median',
    description='B Condition:'
)

c_cond = widgets.Dropdown(
    options=['None', 'positive', 'negative', 'nonzero', 'above_mean', 'above_median'],
    value='None',
    description='C Condition (if present):'
)

wavelet_dropdown = widgets.Dropdown(
    options=['haar', 'db4', 'sym4', 'coif1'],
    value='haar',
    description='Wavelet Family:'
)

run_button = widgets.Button(description="Load Excel & Compute", button_style='success')

output = widgets.Output()

def on_run_clicked(b):
    with output:
        output.clear_output()
        try:
            events_raw = load_events_from_excel(path_input.value)
            conditions = {
                'Event A': a_cond.value if a_cond.value != 'None' else None,
                'Event B': b_cond.value if b_cond.value != 'None' else None,
                'Event C': c_cond.value if c_cond.value != 'None' else None
            }
            result = entanglement_index_multi(
                excel_path=path_input.value,
                conditions=conditions,
                wavelet_family=wavelet_dropdown.value,
                adaptive_decay=True
            )
            print_entanglement_result(result)
        except Exception as e:
            print(f"Error: {str(e)}")

run_button.on_click(on_run_clicked)

display(widgets.VBox([
    widgets.Label("Enter Excel path, select conditions for A/B/C, choose wavelet family, then click Compute:"),
    path_input,
    a_cond,
    b_cond,
    c_cond,
    wavelet_dropdown,
    run_button,
    output
]))