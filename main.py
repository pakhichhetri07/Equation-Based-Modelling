import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -----------------------------
# Input Parameters
# -----------------------------
POP_TOTAL = 1235526.78
n_areas = 58
age_groups = ["Children", "Adults", "Elders"]
n_age = len(age_groups)
SETTINGS = ["home", "school", "office", "park"]

# Total simulation length (Mar 24 2020 -> Oct 10 2020)
days = 200
t_eval = np.linspace(0, days, days + 1)

# Demographic split of population
age_frac = np.array([0.3516, 0.5849, 0.0636])
age_frac = age_frac / age_frac.sum()

# Distribution of population to area and ages (Matrix [58*3])
area_frac = np.ones(n_areas) / n_areas
pop_by_area_age = (POP_TOTAL * area_frac[:, None]) * age_frac[None, :]
pop_by_area_age = pop_by_area_age.astype(float)

# Epidemiological parameters
R0_base = 3.0
infectious_period = 14.0
gamma = 1.0 / infectious_period  # Recovery Rate
incubation_period = 7.0
sigma = 1.0 / incubation_period  # Progression Rate

# mortality rate per day
mortality_by_age_orig = np.array([0.000003, 0.000275, 0.03453])

susceptibility = np.array([0.6, 1.0, 1.2])

# Base contact matrix (rows = infector age gp, cols = susceptible age gp)
contact_matrix_age = np.array([
    [15.0, 8.0, 3.0],
    [8.0, 12.0, 4.0],
    [3.0, 4.0, 6.0]
], dtype=float)

# area mobility
area_mobility = np.full((n_areas, n_areas), 0.05, dtype=float)
np.fill_diagonal(area_mobility, 0.85)
area_mobility = area_mobility / area_mobility.sum(axis=1, keepdims=True)

# Transmission rate calculations: βbase = R0base/spectral_radius(NextGenMatrix) 
contact_susceptibility = contact_matrix_age * susceptibility[None, :]
next_gen_matrix = contact_susceptibility * infectious_period
eigvals = np.linalg.eigvals(next_gen_matrix)
spectral_radius = np.max(np.real(eigvals))
beta_base = float(R0_base / spectral_radius)

# NPI schedule
npi_schedule = [
    (1, 21, {"lockdown": 0.925, "mask": 1.0,   "quarantine": 0.85}),
    (22, 39, {"lockdown": 0.875, "mask": 0.65, "quarantine": 0.80}),
    (40, 53, {"lockdown": 0.825, "mask": 0.70, "quarantine": 0.75}),
    (54, 67, {"lockdown": 0.775, "mask": 0.70, "quarantine": 0.70}),
    (68, 98, {"lockdown": 0.60,  "mask": 0.72, "quarantine": 0.65}),
    (99, 200, {"lockdown": 0.55, "mask": 0.75, "quarantine": 0.60}),
]

def npi_multiplier_smooth(t, npi_schedule, ramp_days=7.0, eps=1e-9, strict=True):

    if len(npi_schedule) == 0:
        return 1.0

    # if before first phase
    if t < npi_schedule[0][0]:
        return 1.0

    # find current phase index (or last if beyond end)
    k = None
    for i, (s, e, effects) in enumerate(npi_schedule):
        if (t >= s) and (t < e):   # half-open interval [s, e)
            k = i
            break
    if k is None:
        k = len(npi_schedule) - 1

    s_k, e_k, effects_k = npi_schedule[k]
    prev_effects = npi_schedule[k - 1][2] if k > 0 else {}

    # blending window start (ramp_days before s_k)
    t_start_blend = s_k - ramp_days

    # fully previous phase
    if t < t_start_blend:
        return _safe_product(prev_effects, eps, strict)

    # blending factor
    if ramp_days <= 0:
        alpha = 1.0 if t >= s_k else 0.0
    else:
        alpha = np.clip((t - t_start_blend) / ramp_days, 0.0, 1.0)

    # interpolate between prev and current effects in log-space
    return _blend_effects(prev_effects, effects_k, alpha, eps, strict)


def _safe_product(effects, eps, strict):
    """Multiply effects safely (all >0)."""
    prod = 1.0
    for k, v in effects.items():
        if strict and v <= 0:
            raise ValueError(f"Effect {k} has non-positive value {v}")
        prod *= max(v, eps)
    return float(prod)


def _blend_effects(prev, curr, alpha, eps, strict):
    """Geometric interpolation between prev and curr effect dicts."""
    all_keys = set(prev.keys()) | set(curr.keys())
    prod = 1.0
    for k in all_keys:
        pv = prev.get(k, 1.0)
        cv = curr.get(k, 1.0)
        if strict and (pv <= 0 or cv <= 0):
            raise ValueError(f"Effect {k} has non-positive values (prev={pv}, curr={cv})")
        pv = max(pv, eps)
        cv = max(cv, eps)
        val = np.exp((1 - alpha) * np.log(pv) + alpha * np.log(cv))
        prod *= val
    return float(prod)

# ------------------------------------------------------------------------------
# IFR (infection fatality ratio) probabilities (0<=p<1) p = mu / (gamma + mu) to per-day mortality rate (μ)
# ------------------------------------------------------------------------------
def ifr_to_mu(ifr_array, gamma_val):
    ifr = np.asarray(ifr_array, dtype=float)
    if np.any((ifr < 0) | (ifr >= 1.0)):
        raise ValueError("IFRs must be in [0,1).")
    mu = (ifr * gamma_val) / (1.0 - ifr)
    return mu

# ----------------------------------------------------------
# Mortality Rate Calculation over the whole infection
# ----------------------------------------------------------

implied_ifrs_orig = mortality_by_age_orig / (gamma + mortality_by_age_orig)
print("Per-day mortality (mortality_by_age):", mortality_by_age_orig)
print("Probability of dying during whole infectious period (IFR from mortality_by_age):", implied_ifrs_orig)

# Children: 0.01% ; Adults: 0.5%; Elders: 5%)
target_ifrs = np.array([0.00001, 0.005, 0.05])

# Convert to per-day mu and assign to mortality_by_age used by ODE
mortality_by_age = ifr_to_mu(target_ifrs, gamma)
print("Target Infection Fatality Ratio (IFRs):", target_ifrs)
print("Mortality Rate during whole Infection (μ):", mortality_by_age)

# -----------------------------
# Compartments and indexing
# -----------------------------
compartments = ["S", "E", "I", "R", "D"]
n_comp = len(compartments)
state_size = n_areas * n_age * n_comp

def idx(a, g, comp):
    comp_idx = compartments.index(comp)
    return (a * n_age + g) * n_comp + comp_idx

# -----------------------------
# Initial conditions
# -----------------------------
initial_infected = 1
y0 = np.zeros(state_size, dtype=float)

# Distribute susceptibles according to pop_by_area_age
for a in range(n_areas):
    for g in range(n_age):
        y0[idx(a, g, "S")] = pop_by_area_age[a, g]

# Place the single initial infected into area 0, split by age (70% adults, 20% children, 10% elders)
infect_adults = initial_infected * 0.7
infect_children = initial_infected * 0.2
infect_elders = initial_infected * 0.1

# initializes the outbreak
def allocate_infection(area, g, amount):
    avail = y0[idx(area, g, "S")]
    to_move = min(avail, amount)
    avail -= to_move
    y0[idx(area, g, "I")] += to_move

allocate_infection(0, 1, infect_adults)   # adults
allocate_infection(0, 0, infect_children) # children
allocate_infection(0, 2, infect_elders)   # elders

# ---------------------------------------------------------------------------------------
# Per-setting contact matrices (rows = infector_age gp, cols = susceptible_age gp)
# ---------------------------------------------------------------------------------------
setting_scaler = {
    "home": 1.2,
    "school": 1.5,
    "office": 1.0,
    "park": 0.5
}
contact_matrix_settings = {}
for s in SETTINGS:
    contact_matrix_settings[s] = contact_matrix_age * setting_scaler[s]

# Per-setting transmissibility factor
setting_beta_factor = {"home": 1.0, "school": 0.8, "office": 0.9, "park": 0.4}

 # age-specific contact multipliers for different locations/Settings
age_mobility_scaler = {
    "home":   {"Children":1.0, "Adults":1.0, "Elders":1.0},
    "school": {"Children":1.2, "Adults":0.1, "Elders":0.0},
    "office": {"Children":0.0, "Adults":1.0, "Elders":0.2},
    "park":   {"Children":0.8, "Adults":0.7, "Elders":0.5},
}
# probability transition matrices between geographic areas.describing how people of different age groups move across areas in different settings
area_mobility_settings = {}
for s in SETTINGS:
    area_mobility_settings[s] = {}
    for age_index, age_name in enumerate(age_groups):
        M = area_mobility * age_mobility_scaler[s][age_name]
        M = M / (M.sum(axis=1, keepdims=True) + 1e-12)
        area_mobility_settings[s][age_index] = M

# -----------------------------
# SEIRD ODE with layered settings
# -----------------------------
def seird_ode(t, y):
    eps = 1e-9
    dydt = np.zeros_like(y, dtype=float)

    # smoothed NPI multiplier (use ramp_days=7 by default)
    beta_t = beta_base * npi_multiplier_smooth(t, npi_schedule, ramp_days=7.0)

    # Build I_ag and N_ag (A x G)
    I_ag = np.zeros((n_areas, n_age), dtype=float)
    N_ag = np.zeros((n_areas, n_age), dtype=float)
    for a in range(n_areas):
        for g in range(n_age):
            I_ag[a, g] = y[idx(a, g, "I")]
            N_ag[a, g] = (
                y[idx(a, g, "S")] +
                y[idx(a, g, "E")] +
                y[idx(a, g, "I")] +
                y[idx(a, g, "R")]
            )

    prevalence = I_ag / (N_ag + eps)  # shape (A, G)

    lam = np.zeros((n_areas, n_age), dtype=float)
    for s in SETTINGS:
        C_s = contact_matrix_settings[s]  # (G, G) rows=infector gp, cols=susceptible g

        # Build MobPrev (A, G) accounting for mobility of each infectious age gp
        MobPrev = np.zeros_like(prevalence)  # (A, G)
        for gp in range(n_age):
            M_gp = area_mobility_settings[s][gp]  # (A, A)
            # prevalence[:, gp] shape (A,), M_gp.dot -> (A,)
            MobPrev[:, gp] = M_gp.dot(prevalence[:, gp])

        # contribution (A, G): sum over gp MobPrev[a,gp] * C_s[gp,g]
        contribution = MobPrev.dot(C_s)  # (A, G)

        factor = beta_t * setting_beta_factor.get(s, 1.0)
        lam += factor * contribution * susceptibility[np.newaxis, :]

    # SEIRD derivatives
    for a in range(n_areas):
        for g in range(n_age):
            S = y[idx(a, g, "S")]
            E = y[idx(a, g, "E")]
            I = y[idx(a, g, "I")]
            R = y[idx(a, g, "R")]

            dS = -lam[a, g] * S
            dE = lam[a, g] * S - sigma * E
            dI = sigma * E - gamma * I - mortality_by_age[g] * I
            dR = gamma * I
            dD = mortality_by_age[g] * I

            dydt[idx(a, g, "S")] = dS
            dydt[idx(a, g, "E")] = dE
            dydt[idx(a, g, "I")] = dI
            dydt[idx(a, g, "R")] = dR
            dydt[idx(a, g, "D")] = dD

    return dydt

# -----------------------------
# Run simulation
# -----------------------------
def run_simulation(t_span=(0, days), t_eval=t_eval, y0=y0):
    sol = solve_ivp(seird_ode, t_span, y0, t_eval=t_eval, vectorized=False, rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError("ODE solver failed: " + str(sol.message))
    return sol

# -----------------------------
# Metrics extraction & plotting
# -----------------------------
def extract_metrics(sol):
    Y = sol.y
    times = sol.t
    total_I = np.zeros(times.shape)
    total_R = np.zeros(times.shape)
    total_D = np.zeros(times.shape)
    total_S = np.zeros(times.shape)
    for ti in range(len(times)):
        Ys = Y[:, ti]
        I_sum = 0.0
        R_sum = 0.0
        D_sum = 0.0
        S_sum = 0.0
        for a in range(n_areas):
            for g in range(n_age):
                I_sum += Ys[idx(a, g, "I")]
                R_sum += Ys[idx(a, g, "R")]
                D_sum += Ys[idx(a, g, "D")]
                S_sum += Ys[idx(a, g, "S")]
        total_I[ti] = I_sum
        total_R[ti] = R_sum
        total_D[ti] = D_sum
        total_S[ti] = S_sum

    peak_idx = np.argmax(total_I)
    peak_size = total_I[peak_idx]
    peak_time = times[peak_idx]
    total_cases = (total_R[-1] + total_D[-1])
    total_deaths = total_D[-1]
    summary = {
        "peak_size": float(peak_size),
        "peak_time_days": float(peak_time),
        "total_cases": float(total_cases),
        "total_deaths": float(total_deaths),
        "time_series": {
            "time": times,
            "I": total_I,
            "R": total_R,
            "D": total_D,
            "S": total_S
        }
    }
    return summary

# Run baseline simulation
sol = run_simulation()
metrics = extract_metrics(sol)

# Print metrics
print(f"Base beta: {beta_base:.6f}")
print(f"Peak infected (total) = {metrics['peak_size']:.0f} persons")
print(f"Peak timing = {metrics['peak_time_days']:.1f} days since start (0=Mar24)")
print(f"Total reported cases (R + D final) = {metrics['total_cases']:.0f}")
print(f"Total deaths = {metrics['total_deaths']:.0f}")

# Age-group breakdown
final = sol.y[:, -1]
age_results = {age: {"cases": 0, "deaths": 0, "population": 0} for age in age_groups}
for g in range(n_age):
    for a in range(n_areas):
        age_results[age_groups[g]]["cases"] += (final[idx(a, g, "R")] + final[idx(a, g, "D")] + final[idx(a, g, "I")])
        age_results[age_groups[g]]["deaths"] += final[idx(a, g, "D")]
        age_results[age_groups[g]]["population"] += pop_by_area_age[a, g]

print("\n=== AGE GROUP BREAKDOWN ===")
for age, data in age_results.items():
    cases = data["cases"]
    deaths = data["deaths"]
    population = data["population"]
    if cases > 0:
        print(f"{age}: {cases:,.0f} cases ({cases/population*100:.3f}% infected), {deaths:,.0f} deaths ({deaths/cases*100:.3f}% CFR)")
    else:
        print(f"{age}: {cases:,.0f} cases (0.0% infected), {deaths:,.0f} deaths (N/A CFR)")

plt.rcParams["font.family"] = "Times New Roman"

ts = metrics['time_series']['time']
I_ts = metrics['time_series']['I']
R_ts = metrics['time_series']['R']
D_ts = metrics['time_series']['D']
S_ts = metrics['time_series']['S']

# Compute daily new infections (incidence) using -ΔS_total (discrete approximation)
incidence = np.zeros_like(S_ts)
incidence[1:] = -np.diff(S_ts)
incidence[incidence < 0] = 0.0  # numerical safety

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# 1. Normalized SEIRD compartments
axes[0].plot(ts, S_ts / POP_TOTAL, label='Susceptible', alpha=0.7)
axes[0].plot(ts, I_ts / POP_TOTAL, label='Infected', color='red')
axes[0].plot(ts, R_ts / POP_TOTAL, label='Recovered', color='green')
axes[0].plot(ts, D_ts / POP_TOTAL, label='Deaths', color='black')
axes[0].set_ylabel('Fraction of population')
axes[0].set_title('SEIRD epidemic dynamics (normalized)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 2. Active infections with NPI shading
npi_labels = [
    "Lockdown Phase 1 (25 Mar - 14 Apr)",
    "Lockdown Phase 2 (15 Apr - 3 May)",
    "Lockdown Phase 3 (4 May - 17 May)",
    "Lockdown Phase 4 (18 May - 31 May)",
    "Unlock 1.0 (1 Jun - 30 Jun)",
    "Unlock 2.0+ (1 Jul - 10 Oct)"
]
phase_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
axes[1].plot(ts, I_ts, color='red', linewidth=2, label='Active Infections')
for i, (start, end, _) in enumerate(npi_schedule):
    if i < len(phase_colors):
        axes[1].axvspan(start, min(end, ts[-1]), alpha=0.15, color=phase_colors[i], label=npi_labels[i])
axes[1].set_ylabel('Active infections')
axes[1].set_title('Active infections with NPI phases')
axes[1].legend(loc='upper right', fontsize=8, frameon=False)
axes[1].grid(True, alpha=0.3)

# 3. Daily incidence (new cases per day)
axes[2].plot(ts, incidence, color='orange', linewidth=2)
axes[2].fill_between(ts, incidence, alpha=0.3)
axes[2].set_ylabel('New cases/day')
axes[2].set_xlabel('Days since Mar 24')
axes[2].set_title('Daily new infections (incidence curve)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
