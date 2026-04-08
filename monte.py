import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy_financial as nf
import time

project_life = 30
tax_rate_base = 0.15       # base tax rate (15%)
discount_rate_base = 0.07

# -------------------------------------------------------
# BASE-CASE VALUES
# -------------------------------------------------------

# CAPEX / FCI
FCI_base = 61_200_000 + 14_000_000   # added 14 million to CAPEX

# Annual revenue (base case)
Revenue_base = 253_837_200

# Scrap value
scrap_base = 0.10 * FCI_base   # pre-tax scrap = 10% of CAPEX

# Annual cost of raw materials at base case (USD/year)
rawmat_cost_base = 186_893_776

# Annual OPEX excluding raw materials (USD/year)
opex_ex_raw_base = 31_124_124

# Annual depreciation (left as in your original model)
depreciation_base = 0.07

# CAPEX split assumptions
capex_split_year1 = 0.40
capex_split_year2 = 0.60

# First production year assumptions
startup_factor = 0.60   # 60% capacity and 60% OPEX in first production year

# -------------------------------------------------------
# BASE CASHFLOW & NPV
# -------------------------------------------------------

def build_after_tax_CF(
    FCI,
    revenue,
    raw_cost,
    opex_ex_raw,
    depreciation,
    scrap,
    tax_rate,
    project_life,
    startup_factor=0.60
):
    """
    Timeline:
    Year 0: no cash flow
    Year 1: 40% CAPEX
    Year 2: 60% CAPEX, no production yet
    Year 3: first production year at 60% capacity and 60% OPEX
    Year 4..(N-1): full operation
    Year N: full operation + scrap
    """

    CF = np.zeros(project_life + 1)

    # CAPEX split over Year 1 and Year 2
    CF[1] = -capex_split_year1 * FCI
    CF[2] = -capex_split_year2 * FCI

    # Year 3: startup year at 60% capacity and 60% OPEX
    startup_revenue = startup_factor * revenue
    startup_raw = startup_factor * raw_cost
    startup_opex_ex = startup_factor * opex_ex_raw

    taxable_income_startup = startup_revenue - startup_raw - startup_opex_ex - depreciation
    tax_startup = tax_rate * taxable_income_startup if taxable_income_startup > 0 else 0.0
    CF[3] = taxable_income_startup - tax_startup + depreciation

    # Years 4 to project_life-1: full operation
    total_opex = raw_cost + opex_ex_raw
    taxable_income = revenue - total_opex - depreciation
    tax = tax_rate * taxable_income if taxable_income > 0 else 0.0
    after_tax_CF = taxable_income - tax + depreciation

    if project_life > 4:
        CF[4:project_life] = after_tax_CF

    # Final year: full operation + scrap
    taxable_income_final = revenue - total_opex - depreciation + scrap
    tax_final = tax_rate * taxable_income_final if taxable_income_final > 0 else 0.0
    CF[project_life] = taxable_income_final - tax_final + depreciation

    return CF


def calc_payback_time(CF):
    cum = 0.0
    went_negative = False

    for t in range(len(CF)):
        prev_cum = cum
        cum += CF[t]

        # Don't count payback until the project has actually gone negative
        if cum < 0:
            went_negative = True

        if went_negative and cum >= 0:
            if CF[t] != 0:
                frac = (0 - prev_cum) / CF[t]
                return (t - 1) + frac
            return float(t)

    return np.nan


# Base-case CF and NPV/IRR check
years = np.arange(project_life + 1)

CF_base = build_after_tax_CF(
    FCI_base,
    Revenue_base,
    rawmat_cost_base,
    opex_ex_raw_base,
    depreciation_base,
    scrap_base,
    tax_rate_base,
    project_life,
    startup_factor=startup_factor
)

NPV_base = np.sum(CF_base / (1 + discount_rate_base) ** years)
IRR_base = nf.irr(CF_base)
base_PBT = calc_payback_time(CF_base)

print(f"\nBase-case NPV            = {NPV_base/1e6:.2f} million USD")
print(f"Base-case IRR            = {IRR_base*100:.2f} %")
print(f"Base-case Payback Time   = {base_PBT:.2f} years")

# ------------------
# MONTE CARLO SETUP
# ------------------

N_SIM = 10000  # Number of simulations

# Capacity factor triangular
capacity_factor = np.random.triangular(0.90, 0.98, 1.00, size=N_SIM)

# Product price triangular
price_factor = np.random.triangular(0.90, 1.00, 1.10, size=N_SIM)

# Combined revenue factor
rev_factor = capacity_factor * price_factor

# Raw materials ±10%
raw_factor = np.random.triangular(0.90, 1.00, 1.10, size=N_SIM)

# OPEX excl raw ±40%
opex_ex_factor = np.random.triangular(0.60, 1.00, 1.40, size=N_SIM)

# CAPEX uncertainty
capex_factor = np.random.lognormal(mean=0, sigma=0.25, size=N_SIM)
capex_factor = 0.9 + (capex_factor - np.mean(capex_factor)) * 0.5

# Arrays to store results
NPV = np.zeros(N_SIM)
IRR = np.full(N_SIM, np.nan)
PI  = np.full(N_SIM, np.nan)
PBT = np.full(N_SIM, np.nan)

# -----------------
# PROGRESS TRACKER
# -----------------

def print_progress(i, total, start_time, last_printed_pct):
    pct = int((i / total) * 100)
    if pct > last_printed_pct:
        elapsed = time.time() - start_time
        if i > 0:
            est_total = elapsed * total / i
            remaining = est_total - elapsed
        else:
            remaining = np.nan

        print(
            f"\rProgress: {pct:3d}% | "
            f"Elapsed: {elapsed:7.1f}s | "
            f"ETA: {remaining:7.1f}s",
            end="",
            flush=True
        )
        return pct
    return last_printed_pct

# -----------------
# MONTE CARLO LOOP
# -----------------

start_time = time.time()
last_printed_pct = -1

for i in range(N_SIM):

    # Sampled parameters
    R = Revenue_base * rev_factor[i]
    raw_cost = rawmat_cost_base * raw_factor[i]
    opex_ex = opex_ex_raw_base * opex_ex_factor[i]
    FCI = FCI_base * capex_factor[i]
    scrap = 0.10 * FCI
    dep = depreciation_base

    # Build scenario cash flow
    CF_i = build_after_tax_CF(
        FCI,
        R,
        raw_cost,
        opex_ex,
        dep,
        scrap,
        tax_rate_base,
        project_life,
        startup_factor=startup_factor
    )

    # NPV
    NPV_i = np.sum(CF_i / (1 + discount_rate_base) ** years)
    NPV[i] = NPV_i

    # IRR
    try:
        IRR[i] = nf.irr(CF_i)
    except Exception:
        IRR[i] = np.nan

    # Profitability Index
    pv_inflows = np.sum(CF_i[CF_i > 0] / (1 + discount_rate_base) ** years[CF_i > 0])
    pv_outflows = -np.sum(CF_i[CF_i < 0] / (1 + discount_rate_base) ** years[CF_i < 0])
    PI[i] = pv_inflows / pv_outflows if pv_outflows > 0 else np.nan

    # Payback Time
    PBT[i] = calc_payback_time(CF_i)

    # Progress update
    last_printed_pct = print_progress(i + 1, N_SIM, start_time, last_printed_pct)

print()  # newline after progress tracker

# --------------
# SUMMARY STATS
# --------------

mean_NPV = np.mean(NPV)
prob_positive = np.mean(NPV > 0)

corr_rev     = np.corrcoef(rev_factor, NPV)[0, 1]
corr_raw     = np.corrcoef(raw_factor, NPV)[0, 1]
corr_opex_ex = np.corrcoef(opex_ex_factor, NPV)[0, 1]
corr_capex   = np.corrcoef(capex_factor, NPV)[0, 1]

mean_IRR = np.nanmean(IRR)
median_IRR = np.nanpercentile(IRR, 50)
mean_PBT = np.nanmean(PBT)
mean_PI = np.nanmean(PI)

print("\n--- Monte Carlo results ---")
print(f"Mean NPV       = ${mean_NPV/1e6:.2f} M")
print(f"P(NPV > 0)     = {prob_positive*100:.1f} %")
print("Correlation with NPV:")
print(f"  Revenue factor       : {corr_rev:.3f}")
print(f"  Raw material factor  : {corr_raw:.3f}")
print(f"  OPEX excl raw factor : {corr_opex_ex:.3f}")
print(f"  CAPEX factor         : {corr_capex:.3f}")

print("\n--- Financial metrics ---")
print(f"Mean IRR       = {mean_IRR*100:.2f} %")
print(f"Median IRR     = {median_IRR*100:.2f} %")
print(f"Mean Payback   = {mean_PBT:.2f} years")
print(f"Mean PI        = {mean_PI:.2f}")

# ------
# PLOTS
# ------

NPV_M = NPV / 1e6

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(NPV_M, bins=40, edgecolor='black', alpha=0.7)
plt.axvline(mean_NPV/1e6, linestyle='--', label=f"Mean = {mean_NPV/1e6:.1f} M")
plt.xlabel("NPV [million USD]")
plt.ylabel("Frequency")
plt.title("Monte Carlo NPV Distribution")
plt.legend()
plt.tight_layout()

# CDF
NPV_sorted = np.sort(NPV_M)
cum_prob = np.linspace(0, 1, N_SIM)

plt.figure(figsize=(8, 5))
plt.plot(NPV_sorted, cum_prob, linewidth=2)
plt.axvline(0, linestyle='--', color='red', label="NPV = 0")
plt.axvline(mean_NPV/1e6, linestyle='--', color='green', label="Mean case NPV")
plt.xlabel("NPV [million USD]")
plt.ylabel("Cumulative probability")
plt.title("Cumulative Distribution of NPV (Monte Carlo)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()