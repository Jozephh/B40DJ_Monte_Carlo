import numpy as np
import matplotlib.pyplot as plt
import numpy_financial as nf
import time
from matplotlib.collections import LineCollection
from matplotlib import colors

project_life = 30
tax_rate_base = 0.15
discount_rate_base = 0.07

# -------------
# INPUT VALUES
# -------------

FCI_base = 75_331_279
Revenue_base = 247_199_320
scrap_base = 0.10 * FCI_base
rawmat_cost_base = 186_893_776
opex_ex_raw_base = 32_030_755

capex_split_year1 = 0.40
capex_split_year2 = 0.60
startup_factor = 0.60

# Depreciation rate
depreciation_rate = 1 - (scrap_base / FCI_base) ** (1 / project_life)

print(f"Calculated depreciation rate = {depreciation_rate * 100:.4f}% per year")

# ----------------
# CASH FLOW MODEL
# ----------------

def build_after_tax_CF(
    FCI,
    revenue,
    raw_cost,
    opex_ex_raw,
    depreciation_rate,
    scrap,
    tax_rate,
    project_life,
    startup_factor=0.60
):
    CF = np.zeros(project_life + 1)
    depreciation_schedule = np.zeros(project_life + 1)
    book_values = np.zeros(project_life + 1)

    book_value = FCI

    CF[1] = -capex_split_year1 * FCI
    CF[2] = -capex_split_year2 * FCI

    for t in range(3, project_life + 1):
        if t < project_life:
            depreciation = depreciation_rate * book_value
            if (book_value - depreciation) < scrap:
                depreciation = book_value - scrap
        else:
            depreciation = book_value - scrap

        depreciation_schedule[t] = depreciation
        book_value -= depreciation
        book_values[t] = book_value

    startup_revenue = startup_factor * revenue
    startup_raw = startup_factor * raw_cost
    startup_opex_ex = startup_factor * opex_ex_raw
    depr_startup = depreciation_schedule[3]

    taxable_income_startup = startup_revenue - startup_raw - startup_opex_ex - depr_startup
    tax_startup = tax_rate * taxable_income_startup if taxable_income_startup > 0 else 0.0
    CF[3] = taxable_income_startup - tax_startup + depr_startup

    total_opex = raw_cost + opex_ex_raw

    for t in range(4, project_life):
        depr = depreciation_schedule[t]
        taxable_income = revenue - total_opex - depr
        tax = tax_rate * taxable_income if taxable_income > 0 else 0.0
        CF[t] = taxable_income - tax + depr

    depr_final = depreciation_schedule[project_life]
    taxable_income_final = revenue - total_opex - depr_final + scrap
    tax_final = tax_rate * taxable_income_final if taxable_income_final > 0 else 0.0
    CF[project_life] = taxable_income_final - tax_final + depr_final

    return CF


def calc_payback_time(CF):
    cum = 0.0
    went_negative = False

    for t in range(len(CF)):
        prev_cum = cum
        cum += CF[t]

        if cum < 0:
            went_negative = True

        if went_negative and cum >= 0:
            if CF[t] != 0:
                frac = (0 - prev_cum) / CF[t]
                return (t - 1) + frac
            return float(t)

    return np.nan


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


# ------------------
# MONTE CARLO SETUP
# ------------------

N_SIM = 100000
years = np.arange(project_life + 1)

capacity_factor = np.random.triangular(0.90, 0.98, 1.00, size=N_SIM)
price_factor = np.random.triangular(0.90, 1.00, 1.10, size=N_SIM)
rev_factor = capacity_factor * price_factor

raw_factor = np.random.triangular(0.90, 1.00, 1.10, size=N_SIM)
opex_ex_factor = np.random.triangular(0.90, 1.00, 1.10, size=N_SIM)

capex_factor = np.random.lognormal(mean=0, sigma=0.25, size=N_SIM)
capex_factor = capex_factor / np.mean(capex_factor)

tax_rate_sim = np.random.triangular(0.10, 0.15, 0.25, size=N_SIM)
discount_rate_sim = np.random.triangular(0.05, 0.07, 0.10, size=N_SIM)

NPV = np.zeros(N_SIM)
IRR = np.full(N_SIM, np.nan)
PI = np.full(N_SIM, np.nan)
PBT = np.full(N_SIM, np.nan)

# -----------------------------------------
# STORE A SUBSET OF PATHS FOR LINE PLOTTING
# -----------------------------------------
# We plot only a sample of paths for speed/clarity
N_PATHS_PLOT = 1000
plot_idx = np.random.choice(N_SIM, size=N_PATHS_PLOT, replace=False)
plot_idx_set = set(plot_idx)

cum_dcf_paths = np.zeros((N_PATHS_PLOT, project_life + 1))
final_npv_paths = np.zeros(N_PATHS_PLOT)

plot_counter = 0

# -----------------
# MONTE CARLO LOOP
# -----------------

start_time = time.time()
last_printed_pct = -1

for i in range(N_SIM):
    R = Revenue_base * rev_factor[i]
    raw_cost = rawmat_cost_base * raw_factor[i]
    opex_ex = opex_ex_raw_base * opex_ex_factor[i]
    FCI = FCI_base * capex_factor[i]
    scrap = 0.10 * FCI
    tax_rate_i = tax_rate_sim[i]
    discount_rate_i = discount_rate_sim[i]

    CF_i = build_after_tax_CF(
        FCI=FCI,
        revenue=R,
        raw_cost=raw_cost,
        opex_ex_raw=opex_ex,
        depreciation_rate=depreciation_rate,
        scrap=scrap,
        tax_rate=tax_rate_i,
        project_life=project_life,
        startup_factor=startup_factor
    )

    NPV_i = np.sum(CF_i / (1 + discount_rate_i) ** years)
    NPV[i] = NPV_i

    try:
        IRR[i] = nf.irr(CF_i)
    except Exception:
        IRR[i] = np.nan

    pv_inflows = np.sum(CF_i[CF_i > 0] / (1 + discount_rate_i) ** years[CF_i > 0])
    pv_outflows = -np.sum(CF_i[CF_i < 0] / (1 + discount_rate_i) ** years[CF_i < 0])
    PI[i] = pv_inflows / pv_outflows if pv_outflows > 0 else np.nan

    PBT[i] = calc_payback_time(CF_i)

    # Save sampled cumulative discounted cash-flow path
    if i in plot_idx_set:
        discounted_cf = CF_i / (1 + discount_rate_i) ** years
        cum_dcf = np.cumsum(discounted_cf)

        cum_dcf_paths[plot_counter, :] = cum_dcf / 1e6   # million USD
        final_npv_paths[plot_counter] = NPV_i / 1e6      # million USD
        plot_counter += 1

    last_printed_pct = print_progress(i + 1, N_SIM, start_time, last_printed_pct)

print()

# --------------
# SUMMARY STATS
# --------------

mean_NPV = np.mean(NPV)
prob_positive = np.mean(NPV > 0)

P10 = np.percentile(NPV, 10)
P25 = np.percentile(NPV, 25)
P50 = np.percentile(NPV, 50)
P75 = np.percentile(NPV, 75)
P90 = np.percentile(NPV, 90)

corr_rev = np.corrcoef(rev_factor, NPV)[0, 1]
corr_raw = np.corrcoef(raw_factor, NPV)[0, 1]
corr_opex_ex = np.corrcoef(opex_ex_factor, NPV)[0, 1]
corr_capex = np.corrcoef(capex_factor, NPV)[0, 1]
corr_tax = np.corrcoef(tax_rate_sim, NPV)[0, 1]
corr_discount = np.corrcoef(discount_rate_sim, NPV)[0, 1]

mean_IRR = np.nanmean(IRR)
median_IRR = np.nanpercentile(IRR, 50)
mean_PBT = np.nanmean(PBT)
mean_PI = np.nanmean(PI)

print("\n--- Monte Carlo results ---")
print(f"Mean NPV       = ${mean_NPV/1e6:.2f} M")
print(f"P(NPV > 0)     = {prob_positive*100:.1f} %")
print(f"50% NPV range  = ${P25/1e6:.2f} M to ${P75/1e6:.2f} M")
print(f"80% NPV range  = ${P10/1e6:.2f} M to ${P90/1e6:.2f} M")

print("Correlation with NPV:")
print(f"  Revenue factor       : {corr_rev:.3f}")
print(f"  Raw material factor  : {corr_raw:.3f}")
print(f"  OPEX excl raw factor : {corr_opex_ex:.3f}")
print(f"  CAPEX factor         : {corr_capex:.3f}")
print(f"  Tax rate             : {corr_tax:.3f}")
print(f"  Discount rate        : {corr_discount:.3f}")

print("\n--- Financial metrics ---")
print(f"Mean IRR       = {mean_IRR*100:.2f} %")
print(f"Median IRR     = {median_IRR*100:.2f} %")
print(f"Mean Payback   = {mean_PBT:.2f} years")
print(f"Mean PI        = {mean_PI:.2f}")

# ------
# PLOTS
# ------

NPV_M = NPV / 1e6
NPV_sorted = np.sort(NPV_M)
cum_prob = np.linspace(0, 1, N_SIM)

P10_M = np.percentile(NPV_M, 10)
P25_M = np.percentile(NPV_M, 25)
P50_M = np.percentile(NPV_M, 50)
P75_M = np.percentile(NPV_M, 75)
P90_M = np.percentile(NPV_M, 90)
mean_M = mean_NPV / 1e6

# ------------------
# CDF – PERCENTILES
# ------------------

plt.figure(figsize=(8, 5))
plt.plot(NPV_sorted, cum_prob, linewidth=2)

plt.axvline(P10_M, linestyle='--', color='red', label="P10–P90 (Risk Range)")
plt.axvline(P90_M, linestyle='--', color='red')
plt.axvline(P25_M, linestyle='--', color='orange', label="P25–P75 (Typical Range)")
plt.axvline(P75_M, linestyle='--', color='orange')
plt.axvline(P50_M, linestyle='--', color='green', label="P50 (Median)")

plt.xlabel("NPV [million USD]")
plt.ylabel("Cumulative probability")
plt.title("CDF of NPV (Percentile Ranges)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# --------------------
# CDF – DECISION VIEW
# --------------------

plt.figure(figsize=(8, 5))
plt.plot(NPV_sorted, cum_prob, linewidth=2)

plt.axvline(0, linestyle='--', color='red', label="NPV = 0")
plt.axvline(mean_M, linestyle='--', color='green', label=f"Mean = {mean_M:.1f} M")

plt.xlabel("NPV [million USD]")
plt.ylabel("Cumulative probability")
plt.title("CDF of NPV (Decision Metrics)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ------------------------
# HISTOGRAM – MEAN NPV
# ------------------------

plt.figure(figsize=(8, 5))
plt.hist(NPV_M, bins=40, edgecolor='black', alpha=0.7)

plt.axvline(mean_M, linestyle='--', color='green', linewidth=2,
            label=f"Mean = {mean_M:.1f} M")

plt.xlabel("NPV [million USD]")
plt.ylabel("Frequency")
plt.title("Monte Carlo NPV Distribution (Mean)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ----------------------------------------------------------
# TRAJECTORY PLOT:
# RED = MOST LIKELY / CENTRAL PATHS
# BLUE = LESS LIKELY / OUTER PATHS
#
# "Most likely" here is approximated as paths whose FINAL NPV
# is closest to the median final NPV.
# ----------------------------------------------------------

# Sort by final NPV
order = np.argsort(final_npv_paths)
cum_dcf_paths_sorted = cum_dcf_paths[order]
final_npv_sorted = final_npv_paths[order]

# Distance from median final NPV
median_final_npv = np.median(final_npv_sorted)
dist_from_median = np.abs(final_npv_sorted - median_final_npv)

# Normalize distance:
# 0 distance -> most central -> red
# large distance -> outer/tail -> blue
max_dist = np.max(dist_from_median)
if max_dist == 0:
    closeness = np.ones_like(dist_from_median)
else:
    closeness = 1 - (dist_from_median / max_dist)

# Optional contrast shaping to make center redder
closeness = closeness ** 0.9

x = years
segments = []
segment_colors = []

cmap = plt.cm.coolwarm

for j in range(N_PATHS_PLOT):
    y = cum_dcf_paths_sorted[j]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    segments.extend(segs)

    color_j = cmap(closeness[j])
    segment_colors.extend([color_j] * (len(x) - 1))

fig, ax = plt.subplots(figsize=(11, 6.5))

lc = LineCollection(
    segments,
    colors=segment_colors,
    linewidths=1.0,
    alpha=0.35
)
ax.add_collection(lc)

# Add percentile envelopes for readability
median_path = np.median(cum_dcf_paths, axis=0)

plt.plot(years, median_path, color='black', linewidth=2.5, label='Median path')
plt.axhline(0, linestyle='--', color='black', linewidth=1.2, alpha=0.8)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(np.min(cum_dcf_paths_sorted), np.max(cum_dcf_paths_sorted))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Less likely / outer paths  →  blue      |      red  ←  More likely / central paths")

plt.xlabel("Project year")
plt.ylabel("Cumulative discounted cash flow [million USD]")
plt.title("Monte Carlo Projection of Cumulative Discounted Cash Flow")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()