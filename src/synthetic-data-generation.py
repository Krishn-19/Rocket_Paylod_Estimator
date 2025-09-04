# This script helps generate sensible synthetic data which is used to train the model.

import os
import math
import numpy as np
import pandas as pd
from scipy import stats
import json
import warnings
warnings.filterwarnings("ignore")

# Globalising all paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_PATH = os.path.join(DATA_DIR, "rockets_pivoted.xlsx")
OUTPUT_XLSX = os.path.join(DATA_DIR, "synthetic_rockets_pivoted.xlsx")
OUTPUT_CSV = os.path.join(DATA_DIR, "synthetic_rockets_pivoted.csv")
REPORT_JSON = os.path.join(DATA_DIR, "synthetic_generation_report.json")

RANDOM_SEED = 42
N_SYNTH = 18000
G0 = 9.80665
MIN_LIFTOFF_TW = 1.2

np.random.seed(RANDOM_SEED)

def safe_array(x):
    a = np.array(x, dtype=float)
    a = a[np.isfinite(a)]
    return a

def sample_empirical(vals, size=1, default=None, bounds=None):
    vals = safe_array(vals)
    if vals.size >= 5:
        choices = np.random.choice(vals, size=size, replace=True)
        jitter = np.random.normal(0, max(vals.std() * 0.05, 1e-6), size=size)
        out = choices + jitter
        if bounds is not None:
            out = np.clip(out, bounds[0], bounds[1])
        return out if size > 1 else float(out[0])
    elif vals.size > 0:
        choices = np.random.choice(vals, size=size, replace=True)
        jitter = np.random.normal(0, max(vals.std() * 0.05, 1e-6), size=size)
        out = choices + jitter
        if bounds is not None:
            out = np.clip(out, bounds[0], bounds[1])
        return out if size > 1 else float(out[0])
    else:
        if size == 1:
            if default is None:
                raise ValueError("No empirical values and no default provided")
            return float(default)
        else:
            if default is None:
                raise ValueError("No empirical values and no default provided")
            return np.full(size, float(default))

def fit_beta_from_samples(samples):
    s = safe_array(samples)
    s = s[(s > 0) & (s < 1)]
    if s.size < 4:
        return None
    mean = s.mean()
    var = s.var(ddof=1)
    if var <= 0 or mean <= 0 or mean >= 1:
        return None
    tmp = mean * (1 - mean) / var - 1
    a = max(0.5, mean * tmp)
    b = max(0.5, (1 - mean) * tmp)
    return float(a), float(b)

raw = pd.read_excel(INPUT_PATH)
# Keeping original row order/pairs so we can exactly recreate layout
row_pairs = list(zip(raw['Stage'].astype(str), raw['Parameter'].astype(str)))

# Rocket columns in the input (all columns except Stage & Parameter)
meta_cols = ['Stage', 'Parameter']
rocket_cols_in = [c for c in raw.columns if c not in meta_cols]

# Creating a long table (Rocket, Stage, Parameter, Value) to extract empirical distributions easily
long = raw.melt(id_vars=meta_cols, value_vars=rocket_cols_in, var_name='Rocket', value_name='Value')

long['Stage_clean'] = long['Stage'].str.strip()
long['Param_clean'] = long['Parameter'].str.strip()
long['col'] = long['Stage_clean'] + '||' + long['Param_clean']
wide = long.pivot(index='Rocket', columns='col', values='Value')
# Reset rocket index to a column for ease
wide = wide.reset_index().rename(columns={'Rocket': 'rocket_name'})

# Extracting empirical distributions
# Payload to orbits are present in Stage == 'Payload(kg)' and Parameter in {LEO, ISS, SSO, MEO, GEO}
orbit_names = ['LEO', 'ISS', 'SSO', 'MEO', 'GEO']
payload_leo_samples = []
orbit_multiplier = {o: [] for o in orbit_names if o != 'LEO'}

# Also collect per-stage delta-v sums for input rockets when available (to establish a baseline dv)
per_rocket_sum_dv = []

# iterate rockets in wide (which has columns like "Payload(kg)||LEO")
for _, row in wide.iterrows():
    # payload LEO
    col_leo = 'Payload(kg)||LEO'
    if col_leo in wide.columns:
        try:
            v = float(row[col_leo])
            if np.isfinite(v) and v > 0:
                payload_leo_samples.append(v)
        except Exception:
            pass
    # other orbits: compute multiplier relative to LEO when both present
    for o in orbit_names:
        if o == 'LEO':
            continue
        col_o = f'Payload(kg)||{o}'
        if col_o in wide.columns and col_leo in wide.columns:
            try:
                v_o = float(row[col_o])
                v_leo = float(row[col_leo])
                if np.isfinite(v_o) and np.isfinite(v_leo) and v_leo > 0:
                    orbit_multiplier[o].append(max(0.0, v_o / v_leo))
            except Exception:
                pass
    # sum delta-v if stage delta-v columns present
    dv_sum = 0.0
    found_dv = False
    for col in wide.columns:
        if 'delta-v' in col.lower() or 'delta_v' in col.lower():
            # rely on 'Delta-v (m/s)' parameter textual match: our columns are like "1st Stage||Delta-v (m/s)"
            if 'Delta-v' in col or 'Delta-v (m/s)' in col or 'delta-v' in col.lower():
                try:
                    val = float(row[col])
                    if np.isfinite(val):
                        dv_sum += val
                        found_dv = True
                except Exception:
                    pass
    if found_dv:
        per_rocket_sum_dv.append(dv_sum)

# Fallbacks if samples are sparse
if len(payload_leo_samples) == 0:
    # Try to harvest any 'Payload' like columns (wildcard)
    payload_candidates = [c for c in wide.columns if 'payload' in c.lower() and 'leo' in c.lower()]
    if payload_candidates:
        for _, row in wide.iterrows():
            try:
                v = float(row[payload_candidates[0]])
                if np.isfinite(v) and v > 0:
                    payload_leo_samples.append(v)
            except Exception:
                pass

# Compute median baseline total dv for LEO (used as heuristic)
if len(per_rocket_sum_dv) >= 3:
    baseline_total_dv = float(np.median(per_rocket_sum_dv))
else:
    # 9.4 km/s makes a reasonable physical baseline for LEO insertion (incl. losses)
    baseline_total_dv = 9400.0

# Prepare orbit multiplier medians (fall back values)
orbit_median_multiplier = {}
for o in orbit_names:
    if o == 'LEO':
        orbit_median_multiplier[o] = 1.0
    else:
        arr = np.array(orbit_multiplier.get(o, []), dtype=float)
        arr = arr[np.isfinite(arr) & (arr >= 0)]
        if arr.size >= 3:
            orbit_median_multiplier[o] = float(np.median(arr))
        elif arr.size >= 1:
            orbit_median_multiplier[o] = float(arr.mean())
        else:
            # fallback educated guesses (based on typical chemical rockets):
            # ISS ~ LEO (slightly lower, similar), SSO ~ 60-80% of LEO, MEO ~ 50-70%, GEO ~ 20-40%
            if o == 'ISS':
                orbit_median_multiplier[o] = 0.9
            elif o == 'SSO':
                orbit_median_multiplier[o] = 0.7
            elif o == 'MEO':
                orbit_median_multiplier[o] = 0.6
            elif o == 'GEO':
                orbit_median_multiplier[o] = 0.35
            else:
                orbit_median_multiplier[o] = 0.5

# Preparing distributions for stage sampling 
# We'll collect per-stage arrays keyed by stage label strings existing in the input (e.g. '1st Stage', '2nd Stage', 'Transfer Stage')
stage_labels = raw['Stage'].unique().tolist()
# Exclude payload stage from stage_labels for stage generation logic
stage_labels_no_payload = [s for s in stage_labels if 'payload' not in s.lower()]

# Build dict: stage -> {isp_vals, dv_vals, s_vals, thrust_vals, impulse_vals}
dists = {}
for st in stage_labels_no_payload:
    d = {'isp': [], 'dv': [], 's_vals': [], 'thrust': [], 'impulse': []}
    # search for relevant parameter columns in wide (col format 'Stage||Parameter')
    for col in wide.columns:
        if isinstance(col, str) and col.startswith(st + '||'):
            param = col.split('||', 1)[1].strip().lower()
            try:
                colvals = safe_array(wide[col].values)
            except Exception:
                colvals = np.array([])
            if 'isp' in param:
                d['isp'].extend(colvals.tolist())
            elif 'delta' in param:
                d['dv'].extend(colvals.tolist())
            elif 'dry mass' in param or 'dry_mass' in param or 'dry mass (kg)' in param.lower():
                # store for structural fraction computation later
                d.setdefault('dry_mass', []).extend(colvals.tolist())
            elif 'start mass' in param or 'start_mass' in param or 'start mass (kg)' in param.lower():
                d.setdefault('start_mass', []).extend(colvals.tolist())
            elif 'propellant' in param:
                d['propellant'] = d.get('propellant', []) + colvals.tolist()
            elif 'thrust' in param:
                d['thrust'].extend(colvals.tolist())
            elif 'impulse' in param:
                d['impulse'].extend(colvals.tolist())
    # structural fraction if dry & start available
    svals = []
    if 'dry_mass' in d and 'start_mass' in d:
        dry = np.array(d['dry_mass'], dtype=float)
        start = np.array(d['start_mass'], dtype=float)
        # elementwise ratios where both present and finite
        # Use pairwise where lengths equal otherwise compute ratio of nonzero means as rough samples
        if dry.size == start.size and dry.size > 0:
            for a, b in zip(dry, start):
                if np.isfinite(a) and np.isfinite(b) and b > 0:
                    r = a / b
                    if 0 < r < 0.5:
                        svals.append(r)
        else:
            # fallback create some ratios from means
            if dry.size > 0 and start.size > 0:
                m_dry = np.nanmedian(dry)
                m_start = np.nanmedian(start)
                if np.isfinite(m_dry) and np.isfinite(m_start) and m_start > 0:
                    r = m_dry / m_start
                    if 0 < r < 0.5:
                        svals.append(r)
    if svals:
        d['s_vals'] = svals
    dists[st] = d

# Define the generator for one rocket and we can call it for multiple later
def generate_one_rocket(dists, payload_leo_samples, orbit_median_multiplier, baseline_total_dv):
    """
    Returns a dict with flattened keys:
      - stage1_start_mass_kg, stage1_final_mass_kg, stage1_dry_mass_kg, stage1_propellant_mass_kg,
        stage1_delta_v_m_s, stage1_isp_s, stage1_total_thrust_n, stage1_total_impulse, stage1_engine_run_time_s, stage1_structural_fraction, stage1_mass_ratio
      - stage2_..., stage3_...
      - payload_leo_kg, payload_iss_kg, payload_sso_kg, payload_meo_kg, payload_geo_kg
      - liftoff_TW, total_initial_mass_kg, sum_delta_v_m_s
    """
    payload_leo = float(sample_empirical(payload_leo_samples, default=500.0))
    # We'll generate stages from top->bottom by using a canonical order if present:
    # Prefer order: Transfer Stage (top), 2nd Stage, 1st Stage (bottom)
    canonical = []
    for label in ['Transfer Stage', '2nd Stage', '1st Stage']:
        if label in dists:
            canonical.append(label)
    # Determine number of stages: randomly 2 or 3 with approximate empirical bias. If Transfer Stage missing, restrict to 2.
    n_possible = len(canonical)
    if n_possible == 0:
        # fallback standard labels
        canonical = ['1st Stage', '2nd Stage']
        n_possible = 2
    # Decide n stages: choose 2 with 60% prob, 3 with 40% if available
    if n_possible >= 3:
        n_stages = 3 if np.random.rand() > 0.6 else 2
    else:
        n_stages = 2
    # use top n_stages from canonical (top is first in canonical)
    chosen_stages = canonical[:n_stages][::-1]  # reverse -> generate top-down (upper first) then set upper_mass progressively
    upper_mass = 0.0
    stage_results = []
    for st_label in chosen_stages:
        sd = dists.get(st_label, {})
        # sample isp
        isp = float(sample_empirical(sd.get('isp', []), default=300.0, bounds=(150.0, 500.0)))
        # sample dv
        dv = float(sample_empirical(sd.get('dv', []), default=2000.0, bounds=(100.0, 12000.0)))
        # structural fraction
        s_vals = sd.get('s_vals', [])
        if len(s_vals) >= 3:
            beta = fit_beta_from_samples(s_vals)
            if beta:
                a, b = beta
                s = float(stats.beta(a, b).rvs())
                s = float(np.clip(s, 0.02, 0.25))
            else:
                s = float(np.clip(np.random.choice(s_vals) * (1 + np.random.normal(0, 0.05)), 0.02, 0.25))
        elif len(s_vals) >= 1:
            s = float(np.clip(np.random.choice(s_vals) * (1 + np.random.normal(0, 0.05)), 0.02, 0.25))
        else:
            s = float(np.random.beta(2.0, 20.0))  # fallback structural fraction
        # mass ratio
        R = math.exp(dv / (isp * G0))
        # enforce R*s < 1
        attempts = 0
        while R * s >= 0.98 and attempts < 20:
            if np.random.rand() < 0.5:
                isp = min(500.0, isp * 1.05)
            else:
                s = max(0.02, s * 0.9)
            R = math.exp(dv / (isp * G0))
            attempts += 1
        if R * s >= 0.98:
            # reduce dv
            dv = max(100.0, dv * 0.7)
            R = math.exp(dv / (isp * G0))
        denom = 1.0 - R * s
        if denom <= 0:
            return None
        m0 = R * (upper_mass + payload_leo) / denom
        mf = m0 / R
        dry_mass = s * m0
        prop_mass = m0 - dry_mass - upper_mass - payload_leo
        if prop_mass <= 0 or m0 <= 0 or dry_mass <= 0:
            return None
        # thrust & impulse
        thrust = float(sample_empirical(sd.get('thrust', []), default=max(1e5, m0 * G0 * MIN_LIFTOFF_TW)))
        impulse = float(sample_empirical(sd.get('impulse', []), default=thrust * 120.0))
        burn_time = float(impulse / thrust) if thrust > 0 else float(np.random.uniform(10, 300))
        # compute calc dv via rocket equation for check
        calc_dv = isp * G0 * math.log(m0 / mf)
        calc_error = calc_dv - dv
        # max acceleration
        max_acc = thrust / max(m0, 1e-6)
        stage_info = {
            'stage_label': st_label,
            'start_mass_kg': m0,
            'final_mass_kg': mf,
            'dry_mass_kg': dry_mass,
            'propellant_mass_kg': prop_mass,
            'delta_v_m_s': dv,
            'isp_s': isp,
            'mass_ratio': R,
            'structural_fraction': s,
            'total_thrust_n': thrust,
            'total_impulse': impulse,
            'engine_run_time_s': burn_time,
            'calc_error_m_s': calc_error,
            'max_acceleration_m_s2': max_acc
        }
        stage_results.append(stage_info)
        upper_mass = m0  # for next (lower) stage

    # liftoff using bottom stage thrust and total initial mass
    bottom = stage_results[-1]
    total_initial_mass = bottom['start_mass_kg']
    total_thrust = bottom['total_thrust_n']
    liftoff_tw = total_thrust / (total_initial_mass * G0)
    if liftoff_tw < MIN_LIFTOFF_TW:
        scale = (MIN_LIFTOFF_TW * total_initial_mass * G0) / (total_thrust + 1e-9)
        bottom['total_thrust_n'] *= scale
        bottom['total_impulse'] *= scale
        bottom['engine_run_time_s'] = bottom['total_impulse'] / bottom['total_thrust_n'] if bottom['total_thrust_n'] > 0 else bottom['engine_run_time_s']
        liftoff_tw = MIN_LIFTOFF_TW
        stage_results[-1] = bottom

    sum_dv = sum(s['delta_v_m_s'] for s in stage_results)

    # Adjust payload_LEO by how the rocket's sum_dv compares to baseline_total_dv (simple heuristic)
    dv_ratio = min(1.0, sum_dv / max(1.0, baseline_total_dv))
    payload_leo_adj = float(max(0.0, payload_leo * dv_ratio))

    # For other orbits, sample a multiplier near empirical median (small gaussian jitter)
    payloads_by_orbit = {'LEO': payload_leo_adj}
    for o in orbit_names:
        if o == 'LEO':
            continue
        med = orbit_median_multiplier.get(o, 0.5)
        # small jitter around median
        mult = float(max(0.0, np.random.normal(med, 0.05 * med)))
        payloads_by_orbit[o] = payload_leo_adj * mult

    # Flatten output keys. stage numbering: stage1 -> bottom stage, stage2 -> next, stage3 -> top (if exists)
    flat = {}
    # note: stage_results currently top->bottom? We appended top first then lower, but we set upper_mass progressively so
    # the last element corresponds to bottom stage. We'll enumerate reversed so stage1 = bottom as requested.
    for idx, s in enumerate(reversed(stage_results), start=1):
        prefix = f'stage{idx}_'
        for k, v in s.items():
            try:
                flat[prefix + k] = float(v)
            except (ValueError, TypeError):
                flat[prefix + k] = v  # keep as string if it can't be converted
    flat['n_stages'] = len(stage_results)
    flat['liftoff_TW'] = float(liftoff_tw)
    flat['total_initial_mass_kg'] = float(total_initial_mass)
    flat['sum_delta_v_m_s'] = float(sum_dv)
    # payloads
    for o in orbit_names:
        flat[f'payload_{o.lower()}_kg'] = float(payloads_by_orbit.get(o, 0.0))

    return flat

# Now we generate N synthetic rockets
synthetic_flat_list = []
failed = 0
print("Generating synthetic rockets... (this may take a moment)")
for i in range(N_SYNTH):
    out = generate_one_rocket(dists, payload_leo_samples, orbit_median_multiplier, baseline_total_dv)
    if out is None:
        failed += 1
        continue
    synthetic_flat_list.append(out)

print(f"Finished generation: successful={len(synthetic_flat_list)}, failed_attempts={failed}")

# The output needs to be pivoted so it can be same as the input.
# Column names "Rocket 1" .. "Rocket N"
rocket_names_out = [f"Rocket {i+1}" for i in range(len(synthetic_flat_list))]

# We'll create a DataFrame with rows = original input rows (Stage, Parameter) in same order,
# columns = ['Stage','Parameter', Rocket 1, Rocket 2, ...]
out_df = pd.DataFrame(columns=['Stage', 'Parameter'] + rocket_names_out)

# Fill Stage & Parameter columns from original row order
stages_list = [rp[0] for rp in row_pairs]
params_list = [rp[1] for rp in row_pairs]
out_df['Stage'] = stages_list
out_df['Parameter'] = params_list

# Helper: mapping from Stage text to stage index in flattened dict
# We expect Stage labels like '1st Stage', '2nd Stage', 'Transfer Stage', 'Payload(kg)' as in your file.
stage_to_index = {
    '1st Stage': 1,
    '2nd Stage': 2,
    'Transfer Stage': 3,
    # Some files may use lower/upper variants; include a few synonyms
    '1st stage': 1,
    '2nd stage': 2,
    'transfer stage': 3,
    'Payload(kg)': None,  # payload handled separately via Parameter being LEO/ISS/...
    'Payload(kg)': None
}

# Function to map Parameter string to flattened key suffix
def map_param_to_key(param_str):
    p = param_str.strip().lower()
    # Match common parameters based on text fragments
    if 'average isp' in p or 'isp' == p or 'isp' in p:
        return 'isp_s'
    if 'delta' in p:
        return 'delta_v_m_s'
    if 'dry' in p:
        return 'dry_mass_kg'
    if 'engine run' in p or 'engine_run' in p:
        return 'engine_run_time_s'
    if 'final mass' in p:
        return 'final_mass_kg'
    if 'max acceleration' in p or 'max_acceleration' in p:
        return 'max_acceleration_m_s2'
    if 'propellant' in p:
        return 'propellant_mass_kg'
    if 'start mass' in p:
        return 'start_mass_kg'
    if 'total impulse' in p or 'impulse' in p:
        return 'total_impulse'
    if 'total thrust' in p or 'thrust' in p:
        return 'total_thrust_n'
    if 'calculation error' in p:
        return 'calc_error_m_s'
    # Orbit payload rows (LEO, ISS etc) will be handled separately
    if p in ['leo', 'iss', 'sso', 'meo', 'geo']:
        return 'payload_orbit'
    # If unknown, return None to leave NaN
    return None

# Fill each rocket column
for col_idx, rocket_data in enumerate(synthetic_flat_list):
    col_name = rocket_names_out[col_idx]
    col_values = []
    # For each row in out_df (same order as input) determine appropriate value
    for stage, param in zip(out_df['Stage'], out_df['Parameter']):
        val = np.nan
        # Payload rows (Stage contains 'Payload' or param is one of orbit names)
        if 'payload' in stage.lower() or param.strip() in orbit_names:
            # determine which orbit
            p_key = param.strip()
            if p_key in orbit_names:
                key = f'payload_{p_key.lower()}_kg'
                val = rocket_data.get(key, np.nan)
        else:
            # stage-specific numeric fields (map Stage like '1st Stage' to stage number)
            # Determine stage index: we want stage1 -> bottom stage; mapping uses the stage labels we used earlier.
            # If stage label not in mapping, try to detect '1st' or '2nd' substrings
            stage_label = stage.strip()
            stage_index = None
            # try direct mapping
            if stage_label in stage_to_index:
                stage_index = stage_to_index[stage_label]
            else:
                sl = stage_label.lower()
                if '1st' in sl or 'first' in sl:
                    stage_index = 1
                elif '2nd' in sl or 'second' in sl:
                    stage_index = 2
                elif 'transfer' in sl or '3rd' in sl or 'third' in sl:
                    stage_index = 3
            # If stage_index found, attempt to map parameter to key
            key_suffix = map_param_to_key(param)
            if stage_index is not None and key_suffix is not None:
                flat_key = f'stage{stage_index}_{key_suffix}'
                if flat_key in rocket_data:
                    val = rocket_data.get(flat_key, np.nan)
                else:
                    # try alternate keys: some keys may be named e.g. 'total_impulse' without suffix
                    if key_suffix == 'final_mass_kg':
                        # try stage{n}_final_mass_kg
                        val = rocket_data.get(flat_key, np.nan)
                    else:
                        val = rocket_data.get(flat_key, np.nan)
            else:
                val = np.nan
        col_values.append(val)
    # assign to out_df column
    out_df[col_name] = col_values

# Preserve same dtype & formatting: convert numeric columns to floats
for c in rocket_names_out:
    out_df[c] = pd.to_numeric(out_df[c], errors='coerce')

# Save outputs
out_df.to_excel(OUTPUT_XLSX, index=False)
out_df.to_csv(OUTPUT_CSV, index=False)

# Write report
report = {
    'n_requested': N_SYNTH,
    'n_generated': len(synthetic_flat_list),
    'n_failed_attempts': failed,
    'baseline_total_dv_used_m_s': baseline_total_dv,
    'orbit_median_multiplier_used': orbit_median_multiplier
}
with open(REPORT_JSON, 'w') as f:
    json.dump(report, f, indent=2)

print("Saved pivoted synthetic dataset to:", OUTPUT_XLSX)
print("Also saved CSV to:", OUTPUT_CSV)
print("Report written to:", REPORT_JSON)
print("Generation summary:", report)
