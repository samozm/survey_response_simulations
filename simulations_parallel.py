import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass, asdict
import warnings
import re
import polars as pl
import os
import psutil
from scipy.special import stdtrit
from scipy.stats import dunnett
from tqdm import tqdm 
import multiprocessing
import gc
import argparse

# Nuke the corrupted environment variable Pixi tried to set
if "R_HOME" in os.environ:
    del os.environ["R_HOME"]

from pymer4.models import glmer, glm
from pymer4.tidystats import easystats, lmerTest
from rpy2 import robjects

# Suppress warnings
warnings.filterwarnings('ignore', category=sm.tools.sm_exceptions.ConvergenceWarning)
warnings.filterwarnings('error', category=sm.tools.sm_exceptions.PerfectSeparationWarning)
warnings.filterwarnings('ignore', message='Maximum Likelihood optimization failed')
warnings.filterwarnings('ignore', message='overflow encountered')
warnings.filterwarnings('ignore', message='divide by zero encountered')


N_DERMS = [100,1000]
SEVERITY_LEVELS = [1, 2, 3, 4, 5]
N_SIMS = 500  # number of sim
TRUE_COEF = 0.8

RESPONSE_RATES  =  [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] 
MISSING_MECHANISMS = ["MCAR", "MNAR"]
TOT_TESTS = len(MISSING_MECHANISMS) * len(N_DERMS) * len(RESPONSE_RATES) - 2 * len(MISSING_MECHANISMS)  # -2 because we don't do 0.2 or 0.1 for N_DERMS == 100)

DUNNETT_LEVEL = 1 - 0.05 / (len(MISSING_MECHANISMS) * len(N_DERMS)) # bonf. adjust DUNNETT for the fact that we're doing this for 9 scenarios and 2 mechanisms


# Factorial experiment scenarios for dermatologist populations
POPULATION_SCENARIOS = {
    "baseline": {
        "description": "Baseline scenario (current settings)",
        "loc_probs": [0.5, 0.3, 0.2],      # urban, suburban, rural
        "caseload_probs": [0.3, 0.4, 0.3], # low, medium, high
        "years_probs": [0.4, 0.35, 0.25],  # early, mid, late
        "aggr_mean": 0.0,
        "aggr_sd": 0.7
    },
    "urban_heavy": {
        "description": "More urban dermatologists",
        "loc_probs": [0.7, 0.2, 0.1],
        "caseload_probs": [0.3, 0.4, 0.3],
        "years_probs": [0.4, 0.35, 0.25],
        "aggr_mean": 0.0,
        "aggr_sd": 0.7
    },
    "rural_heavy": {
        "description": "More rural dermatologists",
        "loc_probs": [0.2, 0.3, 0.5],
        "caseload_probs": [0.3, 0.4, 0.3],
        "years_probs": [0.4, 0.35, 0.25],
        "aggr_mean": 0.0,
        "aggr_sd": 0.7
    },
    "high_caseload": {
        "description": "More high-caseload dermatologists",
        "loc_probs": [0.5, 0.3, 0.2],
        "caseload_probs": [0.2, 0.3, 0.5],
        "years_probs": [0.4, 0.35, 0.25],
        "aggr_mean": 0.0,
        "aggr_sd": 0.7
    },
    "experienced": {
        "description": "More experienced (late-career) dermatologists",
        "loc_probs": [0.5, 0.3, 0.2],
        "caseload_probs": [0.3, 0.4, 0.3],
        "years_probs": [0.2, 0.3, 0.5],
        "aggr_mean": 0.0,
        "aggr_sd": 0.7
    },
    "high_variability": {
        "description": "Higher variability in aggressiveness",
        "loc_probs": [0.5, 0.3, 0.2],
        "caseload_probs": [0.3, 0.4, 0.3],
        "years_probs": [0.4, 0.35, 0.25],
        "aggr_mean": 0.0,
        "aggr_sd": 1.0  # Increased from 0.7
    },
    "low_variability": {
        "description": "Lower variability in aggressiveness",
        "loc_probs": [0.5, 0.3, 0.2],
        "caseload_probs": [0.3, 0.4, 0.3],
        "years_probs": [0.4, 0.35, 0.25],
        "aggr_mean": 0.0,
        "aggr_sd": 0.4  # Decreased from 0.7
    },
    "more_aggressive": {
        "description": "Generally more aggressive dermatologists",
        "loc_probs": [0.5, 0.3, 0.2],
        "caseload_probs": [0.3, 0.4, 0.3],
        "years_probs": [0.4, 0.35, 0.25],
        "aggr_mean": 0.3,  # Shifted mean
        "aggr_sd": 0.7
    },
    "more_conservative": {
        "description": "Generally more conservative dermatologists",
        "loc_probs": [0.5, 0.3, 0.2],
        "caseload_probs": [0.3, 0.4, 0.3],
        "years_probs": [0.4, 0.35, 0.25],
        "aggr_mean": -0.3,  # Shifted mean
        "aggr_sd": 0.7
    }
}


def logistic(x):
    return 1 / (1 + np.exp(-x))


def justify_aggressiveness_parameters():
    """
    Demonstrates the justification for choosing mean=0.0 and SD=0.7 for aggressiveness.
    
    Rationale:
    1. Mean=0.0: Centers the distribution for interpretability (0 = average aggressiveness)
    2. SD=0.7: Creates meaningful heterogeneity without extreme values
    """
    print("=" * 70)
    print("JUSTIFICATION FOR AGGRESSIVENESS PARAMETERS (mean=0.0, SD=0.7)")
    print("=" * 70)
    
    # Generate sample of aggressiveness values
    sample_agg = np.random.normal(0, 0.7, 10000)
    
    # Show the distribution
    print(f"\n1. DISTRIBUTION OF AGGRESSIVENESS:")
    print(f"   - Mean: {sample_agg.mean():.3f} (centered at 0)")
    print(f"   - SD: {sample_agg.std():.3f}")
    print(f"   - 5th percentile: {np.percentile(sample_agg, 5):.3f}")
    print(f"   - 95th percentile: {np.percentile(sample_agg, 95):.3f}")
    print(f"   - Range (5th-95th): {np.percentile(sample_agg, 95) - np.percentile(sample_agg, 5):.3f}")
    
    # Show impact on recommendation probabilities at different severity levels
    print(f"\n2. IMPACT ON RECOMMENDATION PROBABILITIES:")
    print(f"   (Using w_aggr=0.9, showing range across dermatologists)")
    
    alpha = -1.0
    beta_sev = TRUE_COEF #0.8
    w_aggr = 0.9
    
    agg_low = np.percentile(sample_agg, 5)   # Conservative dermatologist
    agg_avg = 0.0                             # Average dermatologist
    agg_high = np.percentile(sample_agg, 95) # Aggressive dermatologist
    
    print(f"\n   Severity | Conservative (5th) | Average (50th) | Aggressive (95th)")
    print(f"   " + "-" * 65)
    
    for sev in SEVERITY_LEVELS:
        lp_low = alpha + beta_sev * sev + w_aggr * agg_low
        lp_avg = alpha + beta_sev * sev + w_aggr * agg_avg
        lp_high = alpha + beta_sev * sev + w_aggr * agg_high
        
        p_low = logistic(lp_low)
        p_avg = logistic(lp_avg)
        p_high = logistic(lp_high)
        
        print(f"      {sev}     |      {p_low:.3f}         |     {p_avg:.3f}      |      {p_high:.3f}")
    
    # Show contribution to log-odds
    print(f"\n3. CONTRIBUTION TO LOG-ODDS SCALE:")
    print(f"   - Weight on aggressiveness (w_aggr): {w_aggr}")
    print(f"   - Typical contribution (±1 SD): ±{w_aggr * 0.7:.3f}")
    print(f"   - Range (5th to 95th percentile): {w_aggr * agg_low:.3f} to {w_aggr * agg_high:.3f}")
    print(f"   - Total range: {w_aggr * (agg_high - agg_low):.3f}")
    
    print(f"\n4. COMPARISON TO OTHER PREDICTORS:")
    print(f"   - Severity effect per unit: {beta_sev}")
    print(f"   - Location effect range: {0.0 - (-0.4):.1f} (urban to rural)")
    print(f"   - Caseload effect range: {0.3 - (-0.3):.1f} (low to high)")
    print(f"   - Aggressiveness effect range (5th-95th): {w_aggr * (agg_high - agg_low):.3f}")
    print(f"   → Aggressiveness creates substantial but realistic variation")
    
    print(f"\n5. INTERPRETATION:")
    print(f"   - SD=0.7 ensures meaningful heterogeneity between dermatologists")
    print(f"   - At low severity (1), probabilities range from ~{logistic(alpha + beta_sev * 1 + w_aggr * agg_low):.2f} to ~{logistic(alpha + beta_sev * 1 + w_aggr * agg_high):.2f}")
    print(f"   - At high severity (5), probabilities range from ~{logistic(alpha + beta_sev * 5 + w_aggr * agg_low):.2f} to ~{logistic(alpha + beta_sev * 5 + w_aggr * agg_high):.2f}")
    print(f"   - This creates realistic provider variation without extreme values")
    
    print("=" * 70)
    print()


def justify_n_simulations():
    """
    Demonstrates that N_SIMS=500 is sufficient by showing:
    1. Monte Carlo standard error decreases with sqrt(n)
    2. Precision is adequate for detecting meaningful differences
    """
    print("=" * 70)
    print("JUSTIFICATION FOR N_SIMS=500")
    print("=" * 70)
    
    # Theoretical Monte Carlo standard error
    print("\n1. THEORETICAL MONTE CARLO STANDARD ERROR:")
    print("   MC standard error = SD / sqrt(N_SIMS)")
    
    assumed_sd = 0.15  # Typical SD of coefficient estimates across simulations
    n_sims_values = [100, 250, 500, 1000, 2000]
    
    print(f"\n   Assuming SD of estimates ≈ {assumed_sd:.3f}:")
    print(f"   {'N_SIMS':<10} {'MC Std Error':<15} {'95% CI Width':<15}")
    print("   " + "-" * 40)
    for n in n_sims_values:
        mc_se = assumed_sd / np.sqrt(n)
        ci_width = 2 * 1.96 * mc_se  # ±1.96 * SE for 95% CI
        print(f"   {n:<10} {mc_se:.5f}        {ci_width:.5f}")
    
    print("\n   → With N_SIMS=500, MC standard error ≈ 0.0067")
    print("   → This gives very precise estimates of the mean bias")
    
    # Power analysis perspective
    print("\n2. PRECISION FOR DETECTING BIAS:")
    print("   With N_SIMS=500, we can reliably detect:")
    
    mc_se_500 = assumed_sd / np.sqrt(500)
    detectable_bias_05 = 1.96 * mc_se_500  # 95% confidence
    detectable_bias_01 = 2.576 * mc_se_500  # 99% confidence
    
    print(f"   - Bias > {detectable_bias_05:.4f} with 95% confidence")
    print(f"   - Bias > {detectable_bias_01:.4f} with 99% confidence")
    print(f"   → This is adequate for detecting practically meaningful bias")
    
    # Efficiency consideration
    print("\n3. EFFICIENCY CONSIDERATION:")
    print("   - N_SIMS=500 balances precision and computational time")
    print(f"   - Moving from 500→1000 simulations:")
    print(f"     • Doubles computation time")
    print(f"     • Only reduces MC SE by {(1 - 1/np.sqrt(2))*100:.1f}%")
    print(f"     • MC SE improvement: {assumed_sd/np.sqrt(500):.5f} → {assumed_sd/np.sqrt(1000):.5f}")
    print("   → Diminishing returns beyond N_SIMS=500")
    
    print("\n4. CONCLUSION:")
    print("   ✓ MC standard error with N_SIMS=500 is very small (≈0.007)")
    print("   ✓ Sufficient precision to detect meaningful bias")
    print("   ✓ Good balance of precision and computational efficiency")
    
    print("=" * 70)
    print()


def generate_dermatologist_population(random_state=123, scenario="baseline",n_derms=100):
    """
    Generate a population of dermatologists with characteristics.
    
    Parameters:
    -----------
    random_state : int
        Random seed for reproducibility
    scenario : str or dict
        Either a key from POPULATION_SCENARIOS or a custom dict with parameters
    """
    rng = np.random.default_rng(random_state)
    
    # Get scenario parameters
    if isinstance(scenario, str):
        if scenario not in POPULATION_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. Choose from {list(POPULATION_SCENARIOS.keys())}")
        params = POPULATION_SCENARIOS[scenario]
    else:
        params = scenario

    # char of derms 
    loc = rng.choice(["urban", "suburban", "rural"], size=n_derms,
                     p=params["loc_probs"])
    caseload = rng.choice(["low", "medium", "high"], size=n_derms,
                          p=params["caseload_probs"])
    years = rng.choice(["early", "mid", "late"], size=n_derms,
                       p=params["years_probs"])

    
    loc_map = {"urban": 0.0, "suburban": -0.2, "rural": -0.4}
    case_map = {"low": -0.3, "medium": 0.0, "high": 0.3}
    year_map = {"early": 0.1, "mid": 0.0, "late": -0.1}

    loc_score = np.vectorize(loc_map.get)(loc)
    case_score = np.vectorize(case_map.get)(caseload)
    year_score = np.vectorize(year_map.get)(years)

    # Dermatologist-level aggressiveness: higher = more likely to recommend
    aggressiveness = rng.normal(loc=params["aggr_mean"], 
                                scale=params["aggr_sd"], 
                                size=n_derms)

    derms = pd.DataFrame({
        "derm_id": np.arange(n_derms),
        "loc": loc,
        "caseload": caseload,
        "years": years,
        "loc_score": loc_score,
        "case_score": case_score,
        "year_score": year_score,
        "aggressiveness": aggressiveness
    })

    return derms


def simulate_true_recommendations(derms, random_state=456):
   
    rng = np.random.default_rng(random_state)

    alpha = -1.0          # baseline intercept (low recommendation at low severity)
    beta_sev = TRUE_COEF  # main effect of severity
    w_loc = 0.3           # urban vs rural differences
    w_case = 0.4          # high caseload more likely to recommend
    w_year = -0.2         # late-career slightly more conservative
    w_aggr = 0.9          # aggressiveness strongly drives recommending
    w_aggr_sev = 0.4      # aggressiveness x severity interaction. more aggressive derms treat more severe cases even more often
    #eps_var = 0.05

    rows = []
    for _, d in derms.iterrows():
        for sev in SEVERITY_LEVELS:
            lp = (alpha
                + beta_sev * sev
                + w_loc * d["loc_score"]
                + w_case * d["case_score"]
                + w_year * d["year_score"]
                + w_aggr * d["aggressiveness"]
                + w_aggr_sev * d["aggressiveness"] * sev)
            #lp = lp + rng.normal(0,np.sqrt(eps_var)) # random error?

            p = logistic(lp)
            recommend = rng.binomial(1, p)

            rows.append({
                "derm_id": d["derm_id"],
                "severity": sev,
                "recommend": recommend,
                "loc": d["loc"],
                "caseload": d["caseload"],
                "years": d["years"],
                "loc_score": d["loc_score"],
                "case_score": d["case_score"],
                "year_score": d["year_score"],
                "aggressiveness": d["aggressiveness"]
            })

    df = pd.DataFrame(rows)
    return df

def fit_glmm_robust(formula, df, full_pop=False, optimizer='bobyqa'):
    """
    Fits a mixed-effects logistic regression using pymer4.
    Includes dynamic gradient parsing and SE checks to ensure robust simulation data.
    """
    # 1. Fit the model with the bobyqa optimizer
    try:
        model = glmer(formula, data=df, family='binomial')
        model.set_factors(["caseload","loc","years"])
        robust_control = f"glmerControl(optimizer={optimizer}, optCtrl=list(maxfun=2e5, relTol=1e-12))"
        model.fit(control=robust_control, verbose=False)
    except Exception as e:
        # Catch hard compilation crashes
        return None, None, None

    # 2. Check for singular fits (random effect variance is zero)
    is_singular = lmerTest.is_singular(model.r_model)
    
    # 3. Interrogate the raw R object for convergence messages
    robjects.globalenv['raw_model'] = model.r_model 
    msgs = robjects.r('raw_model@optinfo$conv$lme4$messages')
    raw_msgs = list(msgs) if msgs != robjects.rinterface.NULL else []
    
    # --- DYNAMIC PYTHON-SIDE EVALUATION ---
    is_acceptable = False 
    
    if not raw_msgs or raw_msgs == ["None"]:
        is_acceptable = True  # Flawless fit
    else:
        for msg in raw_msgs:
            msg_lower = str(msg).lower()
            if "degenerate  hessian" in msg_lower or "degenerate hessian" in msg_lower:
                if full_pop:
                    print("Degenerate Hessian detected (flat likelihood surface). Rejecting.")
                is_acceptable = False
                break
            if "max|grad|" in msg_lower:
                match = re.search(r"max\|grad\|\s*=\s*([0-9.]+)", msg_lower)
                if match:
                    scaled_grad = float(match.group(1))
                    
                    # Condition 1: Perfect (lme4 default)
                    if scaled_grad <= 0.002:
                        is_acceptable = True
                        
                    # Condition 2: The Pragmatic Zone (Check the SE!)
                    elif 0.002 < scaled_grad <= 0.1:
                        try:
                            severity_se = model.result_fit.filter(
                                pl.col("term") == "severity"
                            )["std_error"][0]
                            
                            # Standard error cap for logit models
                            if severity_se < 10.0:
                                is_acceptable = True
                            else:
                                if full_pop:
                                    print(severity_se)
                                is_acceptable = False
                                
                        except Exception:
                            if full_pop: 
                                print(e)
                            is_acceptable = False
                    
                    # Condition 3: The Danger Zone
                    else:
                        is_acceptable = False
                        if full_pop:
                            print(f"scaled_grad={scaled_grad}")
                            print(msg_lower)
                    
                    break # Stop checking once gradient is evaluated
                    
            elif "iteration limit reached" in msg_lower or "maxfun" in msg_lower:
                if full_pop:
                    print("iteration limit reached")
                is_acceptable = False
                break
    
    # --- FINAL EXTRACTION ---
    if is_acceptable and not is_singular:
        coef = model.result_fit.filter(pl.col("term") == "severity")["estimate"][0]
        conf = (model.result_fit.filter(pl.col("term") == "severity")["conf_low"][0], 
                model.result_fit.filter(pl.col("term") == "severity")["conf_high"][0])
        return model, coef, conf
    else:
        if full_pop:
            print(f"is acceptible? {is_acceptable} and is singular? {is_singular}")
        return None, None, None

def fit_logistic_model(df,full_pop=False):
    fmla = "recommend ~ 1 + loc + caseload + years + severity + (1|derm_id)" 
    if not full_pop:
        return fit_glmm_robust(fmla,pl.DataFrame(df),False)
    
    fit1,fit1_coef,fit1_conf = fit_glmm_robust(fmla,pl.DataFrame(df),True)
    optimizers = ["nloptwrap","Nelder_Mead","nlminb","L-BFGS-B","nmkbw","nlminbwrap"]
    n_optimizers = len(optimizers)
    itr = 1
    while fit1 is None and itr < n_optimizers:
        fit1,fit1_coef,fit1_conf = fit_glmm_robust(fmla,pl.DataFrame(df),True,optimizers[itr])
        itr = itr+1
    return fit1,fit1_coef,fit1_conf
        
    
def sample_responders_MCAR(derms, response_rate, rng):
    
    n_resp = int(np.round(response_rate * len(derms)))
    n_resp = max(min(n_resp, len(derms)), 1)
    resp_ids = rng.choice(derms["derm_id"].values, size=n_resp, replace=False)
    return np.sort(resp_ids)


def calibrate_intercept_for_rate(lp_base, target_rate, tol=1e-3, max_iter=50):
   
    low, high = -10.0, 10.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        p = logistic(lp_base + mid)
        mean_p = p.mean()
        if abs(mean_p - target_rate) < tol:
            return mid
        if mean_p < target_rate:
            low = mid
        else:
            high = mid
    return mid  


def sample_responders_MNAR(derms, response_rate, rng):
   
    # Response probability decreased for more aggressive, higher-caseload, 
    # rural, and late-career dermatologists
    a1 = -0.3   # non-urban (more negative loc_score) respond less
    a2 = -0.2   # higher caseload (positive score) respond less
    a3 = -0.2   # late years (score<0) respond less
    a4 = -1.4   # more aggressive treaters respond much less (key MNAR component)

    lp_base = (a1 * derms["loc_score"].values
               + a2 * derms["case_score"].values
               + a3 * derms["year_score"].values
               + a4 * derms["aggressiveness"].values)

    a0 = calibrate_intercept_for_rate(lp_base, response_rate)
    p_resp = logistic(lp_base + a0)
    respond = rng.binomial(1, p_resp)
    resp_ids = derms.loc[respond == 1, "derm_id"].values

    # Fallback to MCAR if by chance we get zero responders
    if len(resp_ids) == 0:
        return sample_responders_MCAR(derms, response_rate, rng)

    return np.sort(resp_ids)



@dataclass
class ScenarioResult:
    mechanism: str
    response_rate: float
    mean_est: float
    true_est: float
    rel_bias_pct: float
    true_coverage: float
    coverage_95: float
    n_successful_fits: int
    n_derms: int
    beta_high: float
    beta_low: float
    beta_var: float
    t_lower: float
    t_upper: float
    dunnett_lower: float
    dunnett_upper: float

@dataclass
class SimResult:
    mechanism: str
    response_rate: float
    est: float
    true_est: float
    rel_bias: float
    lower: float
    upper: float
    sim_idx: int
    n_derms: int

@dataclass
class FactorialResult:
    population_scenario: str
    mechanism: str
    response_rate: float
    mean_est: float
    true_est: float
    rel_bias_pct: float
    coverage_95: float
    n_successful_fits: int
    scenario_description: str
    
@dataclass
class BetaResult:
    betas: list
    population_scenario: str
    mechanism: str
    response_rate: float
    true_beta: float
    
@dataclass
class DunnettResult:
    dunnett_lower: float
    dunnett_upper: float
    population_scenario: str
    mechanism: str
    response_rate: float
    true_beta: float
    
def run_single_sim_wrapper(args):
    """Unpacks a tuple of arguments for imap_unordered."""
    # args looks like: (sim_idx, random_state, scenario, n_derms)
    return run_single_sim(*args)

def run_single_sim(sim_idx, random_state, scenario, n_derms):
    """Worker function to execute a single Monte Carlo iteration."""
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / (1024 * 1024) # MB
    results = []
    failed_fits = 0
    total_fits = 0
    rng = np.random.default_rng(random_state + sim_idx * 11)

    # 1) Generate dermatologist population
    derms = generate_dermatologist_population(random_state=random_state, scenario=scenario, n_derms=n_derms)

    # 2) Simulate full data
    full_data = simulate_true_recommendations(derms, random_state=random_state + 1)

    # 3) Fit "true" logistic model
    _, true_coef, _ = fit_logistic_model(full_data, full_pop=False)
    
    itr = 0
    max_itr = 10
    while true_coef is None and itr < max_itr:
        full_data = simulate_true_recommendations(derms, random_state=random_state + itr + 2)
        _, true_coef, _ = fit_logistic_model(full_data, full_pop=False)
        itr += 1
        
    if true_coef is None:
        return results  # Return empty list if convergence completely fails


    for rr in RESPONSE_RATES:
        for mech in MISSING_MECHANISMS:
            if n_derms == 100 and rr <= 0.2:
                continue
            # Sample responders
            if mech == "MCAR":
                resp_ids = sample_responders_MCAR(derms, rr, rng)
            elif mech == "MNAR":
                resp_ids = sample_responders_MNAR(derms, rr, rng)

            df_resp = full_data[full_data["derm_id"].isin(resp_ids)]

            if df_resp["recommend"].nunique() < 2:
                continue

            _, coef, conf = fit_logistic_model(df_resp)
            total_fits += 1
            if coef is None:
                failed_fits += 1
                continue  
            
            rel_bias = (coef - true_coef) / true_coef * 100.0
            
            sim_result = SimResult(
                mechanism=mech,
                response_rate=rr,
                est=coef,
                true_est=true_coef,
                rel_bias=rel_bias,
                lower=conf[0],
                upper=conf[1],
                sim_idx=sim_idx,
                n_derms=n_derms
            )
            results.append(sim_result)
            
            #if rr == 0.1 and mech == "MCAR":
            #    df_full_100 = full_data[full_data["derm_id"].isin(resp_ids)]
            #    derms_100 = derms[derms["derm_id"].isin(resp_ids)]

    # Handle the subset fitting if df_full_100 was populated
    """ if len(df_full_100) > 0:
        _, true_coef_100, _ = fit_logistic_model(df_full_100, full_pop=True)
        if true_coef_100 is not None:
            for rr in RESPONSE_RATES:
                for mech in MISSING_MECHANISMS:
                    if rr <= 0.2: 
                        continue
                    if mech == "MCAR":
                        resp_ids = sample_responders_MCAR(derms_100, rr, rng)
                    elif mech == "MNAR":
                        resp_ids = sample_responders_MNAR(derms_100, rr, rng)

                    df_resp = df_full_100[df_full_100["derm_id"].isin(resp_ids)]

                    if df_resp["recommend"].nunique() < 2:
                        continue

                    _, coef, conf = fit_logistic_model(df_resp)
                    if coef is None:
                        continue
                    
                    rel_bias = (coef - true_coef_100) / true_coef_100 * 100.0
                    
                    sim_result = SimResult(
                        mechanism=mech,
                        response_rate=rr,
                        est=coef,
                        true_est=true_coef_100,
                        rel_bias=rel_bias,
                        lower=conf[0],
                        upper=conf[1],
                        sim_idx=sim_idx,
                        n_derms=int(n_derms/10)
                    )
                    results.append(sim_result) """
    del full_data 
    del derms
    
    # Force Python garbage collection
    gc.collect()
    
    # Force R garbage collection (Crucial for lme4/pymer4)
    robjects.r('gc()')
    
    #mem_end = process.memory_info().rss / (1024 * 1024)
    
    #if (mem_end - mem_start) > 100:  # Flag if an iteration ate more than 100MB
    #    print(f"Warning: Worker {os.getpid()} leaked {mem_end - mem_start:.1f} MB on sim {sim_idx}")
    return results, failed_fits, total_fits

def run_simulation(random_state=999, scenario="baseline", n_derms=1000):
    """
    Run simulation for a specific population scenario.
    
    Parameters:
    -----------
    random_state : int
        Random seed
    scenario : str
        Population scenario name from POPULATION_SCENARIOS
    """
    
    scenario_desc = POPULATION_SCENARIOS[scenario]["description"] if scenario in POPULATION_SCENARIOS else "Custom"
    print(f"\n[{scenario.upper()}]")

    results_nested = []
    total_failures = 0
    total_fits = 0
    
    # Determine number of workers (leaves 1 core free to prevent OS locking up)
    number_of_cpus = len(psutil.Process().cpu_affinity())
    max_workers = max(1, number_of_cpus - 1)
    print(f"using {max_workers} workers")
    ctx = multiprocessing.get_context('spawn')

    # Workers will completely restart after completing 10 simulations, clearing all RAM
    with ctx.Pool(processes=max_workers, maxtasksperchild=10) as pool:
        # Package your arguments
        tasks = [(sim, random_state, scenario, n_derms) for sim in range(N_SIMS)]
        
        # starmap unpacks the tuples into function arguments
        results_nested = [] #pool.starmap(run_single_sim, tasks)
        pbar = tqdm(
            pool.imap_unordered(run_single_sim_wrapper, tasks),
            total=N_SIMS,
            desc=f"[{scenario}] running sims"
        )
        for res, fails, fits in pbar: 
            results_nested.append(res)
            total_failures += fails
            total_fits += fits
            if total_failures > 0:
                pbar.set_postfix({
                    'failed_fits': str(total_failures) + "/" + str(total_fits)
                })
        
        # Flatten the nested lists
    results = [item for sublist in results_nested for item in sublist]

    results_df = pd.DataFrame([asdict(r) for r in results])
    dunnett_results = []
    
    for mech in MISSING_MECHANISMS:
        all_betas = []
        for rr in RESPONSE_RATES:
            if n_derms == 100 and rr <= 0.2:
                continue
            est_arr        = results_df[(results_df["mechanism"] == mech) & (results_df["response_rate"] == rr) & (results_df["n_derms"] == n_derms)].loc[:,"est"]
            if len(est_arr) == 0: # 0 successful fits!
                print(f"0 successful fits for {mech}, at rate {rr}")
                continue
            true_coef      = results_df[(results_df["mechanism"] == mech) & (results_df["response_rate"] == rr) & (results_df["n_derms"] == n_derms)].loc[:,"true_est"].iloc[0]
            
            beta_result = BetaResult(
                betas=est_arr.tolist(),
                population_scenario=scenario,
                mechanism=mech,
                response_rate=rr,
                true_beta = true_coef
            )
            
            all_betas.append(beta_result)
        true_beta = all_betas[0].true_beta
        n_res     = len(all_betas)
        print(f"n_res={n_res}")
        res = dunnett(*[all_betas[i].betas for i in range(n_res)],
                      control=[true_beta]*len(all_betas[0].betas))
        # Compute 95% Confidence Intervals
        ci = res.confidence_interval(confidence_level=DUNNETT_LEVEL)
        
        i = 0
        for rr in RESPONSE_RATES:
            if n_derms == 100 and rr <= 0.2:
                continue
            dunnett_result = DunnettResult(
                population_scenario=scenario,
                mechanism=mech,
                response_rate=rr,
                dunnett_lower = ci[0][i],
                dunnett_upper = ci[1][i],
                true_beta = true_beta
            )
            dunnett_results.append(dunnett_result)
            i += 1
    
    dunnett_results_df = pd.DataFrame([asdict(r) for r in dunnett_results])
    summary_results = []
    for mech in MISSING_MECHANISMS:
        for rr in RESPONSE_RATES:
            if n_derms == 100 and rr <= 0.2:
                continue
            est_arr        = results_df[(results_df["mechanism"] == mech) & (results_df["response_rate"] == rr) & (results_df["n_derms"] == n_derms)].loc[:,"est"]
            if len(est_arr) == 0: # 0 successful fits!
                print(f"0 successful fits for {mech}, at rate {rr}")
                continue
            lower_arr      = results_df[(results_df["mechanism"] == mech) & (results_df["response_rate"] == rr) & (results_df["n_derms"] == n_derms)].loc[:,"lower"]
            upper_arr      = results_df[(results_df["mechanism"] == mech) & (results_df["response_rate"] == rr) & (results_df["n_derms"] == n_derms)].loc[:,"upper"]
            rel_bias_arr   = results_df[(results_df["mechanism"] == mech) & (results_df["response_rate"] == rr) & (results_df["n_derms"] == n_derms)].loc[:,"rel_bias"]
            true_coef      = results_df[(results_df["mechanism"] == mech) & (results_df["response_rate"] == rr) & (results_df["n_derms"] == n_derms)].loc[:,"true_est"].iloc[0]
            mean_est       = (est_arr - true_coef).mean()
            coverage       = np.mean((lower_arr <= true_coef) & (upper_arr >= true_coef)) * 100.0
            coverage_UNDER = np.mean((lower_arr <= TRUE_COEF) & (upper_arr >= TRUE_COEF)) * 100.0
            beta_low       = np.percentile(est_arr - true_coef, 2.5/TOT_TESTS) 
            beta_high      = np.percentile(est_arr - true_coef, (1-2.5/TOT_TESTS))
            beta_var       = (est_arr - true_coef).var() 
            rel_bias       = rel_bias_arr.mean()
            
            dunnett_lower  = dunnett_results_df[(dunnett_results_df["mechanism"] == mech) & (dunnett_results_df["response_rate"] == rr)].loc[:,"dunnett_lower"].iloc[0]
            dunnett_upper  = dunnett_results_df[(dunnett_results_df["mechanism"] == mech) & (dunnett_results_df["response_rate"] == rr)].loc[:,"dunnett_upper"].iloc[0]
            
            t_lower = mean_est - stdtrit(len(est_arr),1-(0.05/TOT_TESTS))*beta_var
            t_upper = mean_est + stdtrit(len(est_arr),1-(0.05/TOT_TESTS))*beta_var
            print(
                f"[{scenario}] {mech} | response={rr:.1f} | "
                f"true_coef={mean_est:.3f} | rel_bias={rel_bias:.1f}% | "
                f"coverage={coverage:.1f}% | true_coverage={coverage_UNDER:.1f} | "
                f"t interval= ({t_lower:.2f},{t_upper:.2f}) | bootstrap interval= ({beta_low:.2f},{beta_high:.2f})|"
                f"dunnett interval= ({dunnett_lower:.2f},{dunnett_upper:.2f}) |"
                f"n_fits={len(est_arr)}" 
            )

            scenario_result = ScenarioResult(
                mechanism=mech,
                response_rate=rr,
                mean_est=mean_est,
                true_est=true_coef,
                rel_bias_pct=rel_bias,
                coverage_95=coverage,
                true_coverage=coverage_UNDER,
                n_successful_fits=len(est_arr),
                n_derms = n_derms,
                beta_low = beta_low,
                beta_high = beta_high,
                beta_var = beta_var,
                t_lower = t_lower,
                t_upper = t_upper,
                dunnett_lower = dunnett_lower,
                dunnett_upper = dunnett_upper
            )
            summary_results.append(scenario_result)


    # results final 
    summary_results_df = pd.DataFrame([asdict(r) for r in summary_results])
    return summary_results_df


def run_factorial_experiment(scenarios=None, n_derms=1000, random_state=999):
    """
    Run simulations across multiple population scenarios (factorial design).
    
    Parameters:
    -----------
    scenarios : list of str, optional
        List of scenario names to test. If None, tests all scenarios.
    random_state : int
        Base random seed (each scenario gets a different seed)
    
    Returns:
    --------
    pd.DataFrame
        Combined results from all scenarios
    """
    if scenarios is None:
        scenarios = list(POPULATION_SCENARIOS.keys())
    
    save_path = "results/" + "_".join(scenarios) + ".csv"
    
    print("=" * 70)
    print("FACTORIAL EXPERIMENT: TESTING MULTIPLE POPULATION SCENARIOS")
    print("=" * 70)
    print(f"\nScenarios to test: {scenarios}")
    print(f"Response rates: {RESPONSE_RATES}")
    print(f"Missing mechanisms: {MISSING_MECHANISMS}")
    print(f"Simulations per condition: {N_SIMS}")
    print("=" * 70)
    
    all_results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"n_derms={n_derms}")
        print(f"\n{'='*70}")
        print(f"SCENARIO {i+1}/{len(scenarios)}: {scenario}")
        print(f"Description: {POPULATION_SCENARIOS[scenario]['description']}")
        print(f"{'='*70}")
        
        # Run simulation for this scenario with unique seed
        scenario_results = run_simulation(random_state=random_state + i*100,
                                        scenario=scenario, n_derms=n_derms)
        
        # Add scenario information to results
        scenario_results['n_derms'] = n_derms
        scenario_results['population_scenario'] = scenario
        scenario_results['scenario_description'] = POPULATION_SCENARIOS[scenario]['description']
        
        all_results.append(scenario_results)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(save_path, index=False)
    
    print("\n" + "=" * 70)
    print("FACTORIAL EXPERIMENT COMPLETE")
    print("=" * 70)
    
    print(combined_df.iloc[1:5])
    
    return combined_df


def analyze_factorial_results(results_df, save_path=None):
    """
    Analyze and summarize results from factorial experiment.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from run_factorial_experiment
    save_path : str, optional
        Path to save analysis outputs
    """
    print("\n" + "=" * 70)
    print("FACTORIAL EXPERIMENT ANALYSIS")
    print("=" * 70)
    
    # 1. Summary statistics across scenarios
    print("\n1. BIAS COMPARISON ACROSS SCENARIOS (MNAR mechanism)")
    print("   (Showing mean relative bias % at different response rates)")
    print()
    
    mnar_results = results_df[results_df['mechanism'] == 'MNAR'].copy()
    
    # Pivot table for easy comparison
    bias_pivot = mnar_results.pivot_table(
        values='rel_bias_pct',
        index='population_scenario',
        columns='response_rate',
        aggfunc='mean'
    )
    
    print(bias_pivot.round(2))
    
    # 2. Identify scenarios with most bias
    print("\n2. SCENARIOS WITH HIGHEST BIAS (MNAR, low response rates)")
    print()
    
    low_response = mnar_results[mnar_results['response_rate'] <= 0.3]
    worst_bias = low_response.groupby('population_scenario')['rel_bias_pct'].mean().sort_values()
    
    print("   Average bias at response rates ≤ 30%:")
    for scenario, bias in worst_bias.items():
        desc = POPULATION_SCENARIOS[scenario]['description']
        print(f"   {scenario:20s}: {bias:6.2f}% - {desc}")
    
    # 3. Robustness check - variance in bias across scenarios
    print("\n3. ROBUSTNESS: Variance in bias across scenarios")
    print()
    
    bias_by_rate = mnar_results.groupby('response_rate')['rel_bias_pct'].agg(['mean', 'std', 'min', 'max'])
    bias_by_rate['range'] = bias_by_rate['max'] - bias_by_rate['min']
    
    print("   Response Rate | Mean Bias | Std Dev | Min | Max | Range")
    print("   " + "-" * 60)
    for rr, row in bias_by_rate.iterrows():
        print(f"   {rr:13.1f} | {row['mean']:9.2f} | {row['std']:7.2f} | "
              f"{row['min']:6.2f} | {row['max']:6.2f} | {row['range']:6.2f}")
    
    # 4. Coverage comparison
    print("\n4. COVERAGE COMPARISON (95% CI should cover true value)")
    print()
    
    coverage_pivot = mnar_results.pivot_table(
        values='coverage_95',
        index='population_scenario',
        columns='response_rate',
        aggfunc='mean'
    )
    
    print("   Coverage % by scenario and response rate:")
    print(coverage_pivot.round(1))
    
    # 5. Key findings
    print("\n5. KEY FINDINGS:")
    print()
    
    # Find most robust scenario (least variation in bias)
    scenario_bias_std = mnar_results.groupby('population_scenario')['rel_bias_pct'].std()
    most_robust = scenario_bias_std.idxmin()
    
    # Find most sensitive scenario (most variation in bias)
    most_sensitive = scenario_bias_std.idxmax()
    
    print(f"   ✓ Most robust scenario: {most_robust}")
    print(f"     (Least variation in bias across response rates: σ={scenario_bias_std[most_robust]:.2f}%)")
    print()
    print(f"   ✓ Most sensitive scenario: {most_sensitive}")
    print(f"     (Most variation in bias across response rates: σ={scenario_bias_std[most_sensitive]:.2f}%)")
    print()
    
    # Check if bias direction is consistent
    bias_signs = mnar_results.groupby('population_scenario')['rel_bias_pct'].apply(
        lambda x: (x < 0).sum() / len(x) * 100
    )
    
    print(f"   ✓ Consistency of bias direction (% negative bias):")
    for scenario, pct in bias_signs.items():
        print(f"     {scenario:20s}: {pct:5.1f}%")
    
    print("\n" + "=" * 70)
    
    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")
    
    return {
        'bias_pivot': bias_pivot,
        'coverage_pivot': coverage_pivot,
        'scenario_bias_std': scenario_bias_std,
        'most_robust': most_robust,
        'most_sensitive': most_sensitive
    }


if __name__ == "__main__":
    # Always run justifications first
    print("\n" + "="*70)
    print("PARAMETER JUSTIFICATION ANALYSES")
    print("="*70 + "\n")
    
    justify_aggressiveness_parameters()
    justify_n_simulations()
    
    # Then run the full factorial experiment
    print("\n" + "="*70)
    print("FACTORIAL EXPERIMENT")
    print("="*70 + "\n")
    
    # 1. Set up the parser
    parser = argparse.ArgumentParser(description="Run the dermatologist simulation.")
    
    # 2. Add your arguments
    parser.add_argument(
        "--scenario", 
        type=str, 
        nargs="+",
        default="baseline", 
        help="The population scenario to run (e.g., baseline, urban_heavy)"
    )
    
    parser.add_argument(
        "--derms", 
        type=int, 
        default="1000", 
        help="The number of dermatologists to sample"
    )

    # 3. Parse the arguments
    args = parser.parse_args()
    
    # 4. Access them using dot notation
    print(f"Starting simulation for scenario: {args.scenario} with {N_DERMS} derms.")
    
    # Example of plugging it into your existing function:
    # run_simulation(scenario=args.scenario, n_derms=args.n_derms)
    
    factorial_results = run_factorial_experiment(scenarios=args.scenario,n_derms=args.derms,random_state=999*args.derms)
    
    # Analyze results
    analysis = analyze_factorial_results(
        factorial_results[factorial_results['n_derms'] == args.derms], 
        save_path=f"results/factorial_results_"+"_".join(args.scenario)+f"_{args.derms}.csv"
    )
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)