import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
import numpy_financial as nf

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

# Please download the xlsm file and change the file path to match it's location, otherwise the file will not run correctly. 
# Alternatively, copy the values from the file, change the code and input them below.

file_path = r"C:\Users\joe\OneDrive - Heriot-Watt University\Uni_Cloud\Year 4\B40DJ_Sus_Mgmt_&_Process_Econ\Assignment 2\Assignment 2.xlsm"

data0 = pd.read_excel(file_path, sheet_name=0)
data1 = pd.read_excel(file_path, sheet_name=1)
data2 = pd.read_excel(file_path, sheet_name=2)
data3 = pd.read_excel(file_path, sheet_name=3)
data4 = pd.read_excel(file_path, sheet_name=4)
data5 = pd.read_excel(file_path, sheet_name=5)

project_life = 20          
tax_rate_base = 0.20       # base tax rate (20 %)

# -------------------------------------------------------
# 1. BASE-CASE VALUES
# -------------------------------------------------------

# Discount rate (ensure decimal, not 10 for 10 %)
discount_rate_base = float(data4.iloc[1, 17])
if discount_rate_base > 1:
    discount_rate_base /= 100.0

# CAPEX / FCI
FCI_base = float(data4.iloc[2, 4])

# Annual revenue (base case)
Revenue_base = 185500800*0.98 #float(data4.iloc[17, 3]) # 2% hit on product price due to bulk contract

# Sum of equipment cost (for scrap)
Sigma_Ce = float(data4.iloc[2, 2])
scrap_base = 0.10 * Sigma_Ce   # pre-tax scrap = 10% of purchased equipment cost

# Annual cost of raw materials at base case (USD/year)
rawmat_cost_base = 136801884*0.9 #float(data3.iloc[12, 6]) 10% discount on feedstock price

# Annual OPEX excluding raw materials (USD/year)
opex_ex_raw_base = 30290547 #float(data3.iloc[29, 3]) 

# Annual depreciation (same every year, from Excel)
depreciation_base = float(data4.iloc[5, 10])

# -------------------------------------------------------
# 2. NPV FUNCTION
# -------------------------------------------------------

def calc_NPV_after_tax(FCI, revenue, raw_cost, opex_ex_raw,
                       depreciation, scrap, tax_rate, discount_rate,
                       project_life=20):
    """
    After-tax NPV:
      - Year 0: CF0 = -FCI
      - Years 1..(N-1):
          taxable_income = revenue - (raw + other opex) - depreciation
          tax            = tax_rate * taxable_income
          CF             = taxable_income - tax + depreciation
      - Year N:
          taxable_income_final = revenue - (raw + other opex) - depreciation + scrap
          tax_final            = tax_rate * taxable_income_final
          CF_final             = taxable_income_final - tax_final + depreciation
        => CF_final = normal after-tax CF + scrap*(1 - tax_rate)
    """
    years = np.arange(project_life + 1)
    CF = np.zeros(project_life + 1)

    # Year 0
    CF[0] = -FCI

    total_opex = raw_cost + opex_ex_raw

    # --- Years 1..project_life-1 ---
    taxable_income = revenue - total_opex - depreciation
    tax = tax_rate * taxable_income if taxable_income > 0 else 0.0
    after_tax_CF = taxable_income - tax + depreciation

    CF[1:project_life] = after_tax_CF

    # --- Final year: include scrap as taxable ---
    taxable_income_final = revenue - total_opex - depreciation + scrap
    tax_final = tax_rate * taxable_income_final if taxable_income_final > 0 else 0.0
    CF[project_life] = taxable_income_final - tax_final + depreciation

    # NPV
    NPV = np.sum(CF / (1 + discount_rate) ** years)
    return NPV, CF

# Base-case NPV check
NPV_base, CF_base = calc_NPV_after_tax(
    FCI_base, Revenue_base, rawmat_cost_base,
    opex_ex_raw_base, depreciation_base,
    scrap_base, tax_rate, discount_rate
)
print(f"\nBase-case after-tax NPV = {NPV_base/1e6:.2f} million USD")

# -------------------------------------------------------
# 3. MONTE CARLO SETUP
# -------------------------------------------------------

N_SIM = 100000 # Number of simulations
years = np.arange(project_life + 1)

# Uncertainty ranges 
rev_low, rev_high         = 0.6, 1.4   # ±40% on revenue (product price)
raw_low, raw_high         = 0.6, 1.4   # ±40% on raw materials price
opex_ex_low, opex_ex_high = 0.6, 1.4   # ±40% on OPEX excl. raw
capex_low, capex_high     = 0.6, 1.4   # ±40% on FCI and scrap

# Sample random multipliers
rev_factor      = np.random.uniform(rev_low,     rev_high,     N_SIM)
raw_factor      = np.random.uniform(raw_low,     raw_high,     N_SIM)
opex_ex_factor  = np.random.uniform(opex_ex_low, opex_ex_high, N_SIM)
capex_factor    = np.random.uniform(capex_low,   capex_high,   N_SIM)

# Arrays to store results
NPV = np.zeros(N_SIM)
IRR = np.full(N_SIM, np.nan)
PI  = np.full(N_SIM, np.nan)
PBT = np.full(N_SIM, np.nan)

# -------------------------------------------------------
# 4. MONTE CARLO LOOP
# -------------------------------------------------------

for i in range(N_SIM):

    # Apply random factors
    R        = Revenue_base * rev_factor[i]
    raw_cost = rawmat_cost_base * raw_factor[i]
    opex_ex  = opex_ex_raw_base * opex_ex_factor[i]
    FCI      = FCI_base * capex_factor[i]
    scrap    = scrap_base * capex_factor[i]
    dep      = depreciation_base  # keep depreciation fixed

    # Compute after-tax NPV and cash flows for this scenario
    NPV_i, CF_i = calc_NPV_after_tax(
        FCI, R, raw_cost, opex_ex,
        dep, scrap, tax_rate, discount_rate,
        project_life=project_life
    )
    NPV[i] = NPV_i

    # --- IRR ---
    try:
        irr_val = nf.irr(CF_i)
        IRR[i] = irr_val
    except Exception:
        IRR[i] = np.nan

    # --- Profitability Index (PI) ---
    # PI = PV(positive CF from year 1..N) / |initial investment|
    pv_positive = np.sum(CF_i[1:] / (1 + discount_rate) ** years[1:])
    if CF_i[0] < 0:
        PI[i] = pv_positive / (-CF_i[0])
    else:
        PI[i] = np.nan

    # --- Payback Time ---
    cum = 0.0
    pbt_val = np.nan
    for t in range(project_life + 1):
        prev_cum = cum
        cum += CF_i[t]
        if cum >= 0:
            if t == 0:
                pbt_val = 0.0
            else:
                # Linear interpolation between t-1 and t
                if CF_i[t] != 0:
                    frac = (0 - prev_cum) / CF_i[t]
                    pbt_val = (t - 1) + frac
                else:
                    pbt_val = float(t)
            break
    PBT[i] = pbt_val

# -------------------------------------------------------
# 5. SUMMARY STATS
# -------------------------------------------------------

mean_NPV = np.mean(NPV)
prob_positive = np.mean(NPV > 0)

corr_rev     = np.corrcoef(rev_factor,     NPV)[0, 1]
corr_raw     = np.corrcoef(raw_factor,     NPV)[0, 1]
corr_opex_ex = np.corrcoef(opex_ex_factor, NPV)[0, 1]
corr_capex   = np.corrcoef(capex_factor,   NPV)[0, 1]

mean_IRR = np.nanmean(IRR)   # as fraction
p50_IRR = np.nanpercentile(IRR, 50)
mean_PI  = np.nanmean(PI)
mean_PBT = np.nanmean(PBT)

print("\n--- Monte Carlo results ---")
print(f"Mean NPV      = ${mean_NPV/1e6:.2f} M")
print(f"P(NPV > 0)    = {prob_positive*100:.1f} %")
print("Correlation with NPV:")
print(f"Revenue factor:         {corr_rev:.3f}")
print(f"Raw material factor:    {corr_raw:.3f}")
print(f"OPEX excl raw factor:   {corr_opex_ex:.3f}")
print(f"CAPEX factor:           {corr_capex:.3f}")

print("\n--- Extra financial metrics ---")
print(f"Mean IRR      = {mean_IRR*100:.2f} %")
print(f"Median IRR    = {p50_IRR*100:.3f} %")
print(f"Mean PI       = {mean_PI:.3f}")
print(f"Mean Payback Time  = {mean_PBT:.2f} years")

# -------------------------------------------------------
# 6. PLOTS
# -------------------------------------------------------

NPV_M = NPV / 1e6

# (a) Histogram
plt.figure(figsize=(8, 5))
plt.hist(NPV_M, bins=40, edgecolor='black', alpha=0.7)
plt.axvline(mean_NPV/1e6, linestyle='--',
            label=f"Mean = {mean_NPV/1e6:.1f} M")
plt.xlabel("NPV [million USD]")
plt.ylabel("Frequency")
plt.title("Monte Carlo NPV Distribution")
plt.legend()
plt.tight_layout()

# (b) CDF / "all data" view
NPV_sorted = np.sort(NPV_M)
cum_prob = np.linspace(0, 1, N_SIM)

plt.figure(figsize=(8, 5))
plt.plot(NPV_sorted, cum_prob, linewidth=2)
plt.axvline(0, linestyle='--', color = 'red', label="NPV = 0")
plt.axvline(mean_NPV/1e6 ,linestyle='--', color = 'green', label="Mean case NPV")
plt.axvline(FCI_base/1e6, linestyle='--', color = 'orange', label="PI = 1")
plt.xlabel("NPV [million USD]")
plt.ylabel("Cumulative probability")
plt.title("Cumulative Distribution of NPV (Monte Carlo)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
