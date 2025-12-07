import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
file_path = r"C:\Users\joe\OneDrive - Heriot-Watt University\Uni_Cloud\Year 4\B40DJ_Sus_Mgmt_&_Process_Econ\Assignment 2\Assignment 2.xlsm"

data0 = pd.read_excel(file_path, sheet_name=0)
data1 = pd.read_excel(file_path, sheet_name=1)
data2 = pd.read_excel(file_path, sheet_name=2)
data3 = pd.read_excel(file_path, sheet_name=3)
data4 = pd.read_excel(file_path, sheet_name=4)
data5 = pd.read_excel(file_path, sheet_name=5)

project_life = 20
tax_rate = 0.20   # 20% tax

# -------------------------------------------------------
# 1. BASE-CASE VALUES
# -------------------------------------------------------

# Discount rate (make sure this ends up as 0.10, not 10)
discount_rate = float(data4.iloc[1, 17])
if discount_rate > 1:       # if Excel stores 10 for 10%
    discount_rate /= 100.0

# CAPEX / FCI
FCI_base = float(data4.iloc[2, 4])

# Annual revenue (base case)
Revenue_base = float(data4.iloc[17, 3])

# Sum of equipment cost (for scrap)
Sigma_Ce = float(data4.iloc[2, 2])
scrap_base = 0.10 * Sigma_Ce   # pre-tax scrap = 10% of purchased equipment cost

# Annual cost of raw materials at base case (USD/year)
rawmat_cost_base = float(data3.iloc[12, 6])

# Annual OPEX excluding raw materials (USD/year)
opex_ex_raw_base = float(data3.iloc[29, 3])

# Annual depreciation
depreciation_base = float(data4.iloc[5,10])  

print("FCI_base          =", FCI_base)
print("Revenue_base      =", Revenue_base)
print("rawmat_cost_base  =", rawmat_cost_base)
print("opex_ex_raw_base  =", opex_ex_raw_base)
print("depreciation_base =", depreciation_base)
print("scrap_base        =", scrap_base)
print("discount_rate     =", discount_rate)

# -------------------------------------------------------
# 2. NPV FUNCTION WITH TAX
# -------------------------------------------------------

def calc_NPV(FCI, revenue, raw_cost, opex_ex_raw, depreciation, scrap, tax_rate, discount_rate, project_life):
    
    #After-tax NPV:
    #  - Year 0: CF0 = -FCI
    #  - Years 1..(N-1): CF = (rev - costs - dep)*(1-tax) + dep
    #  - Year N: same + after-tax scrap: scrap*(1-tax)

    years = np.arange(project_life + 1)
    CF = np.zeros(project_life + 1)

    # Year 0
    CF[0] = -FCI

    # Total operating cost (raw + other opex)
    total_opex = raw_cost + opex_ex_raw

    # Annual pre-tax profit before depreciation
    taxable_base = revenue - total_opex - depreciation

    # Years 1..project_life-1
    for t in range(1, project_life):
        taxable = taxable_base
        tax = tax_rate * taxable if taxable > 0 else 0.0
        CF[t] = taxable - tax + depreciation   # = (rev - opex - dep)*(1-tax) + dep

    # Final year: same operating CF + after-tax scrap
    taxable = taxable_base
    tax = tax_rate * taxable if taxable > 0 else 0.0
    after_tax_operating = taxable - tax + depreciation
    after_tax_scrap = scrap * (1 - tax_rate)  # assume scrap is taxable gain
    CF[project_life] = after_tax_operating + after_tax_scrap

    # Discounted NPV
    NPV = np.sum(CF / (1 + discount_rate) ** years)
    return NPV

NPV_base = calc_NPV(FCI_base, Revenue_base, rawmat_cost_base,
                    opex_ex_raw_base, depreciation_base,
                    scrap_base, tax_rate, discount_rate, project_life)

print(f"Base-case NPV (after 20% tax) = {NPV_base/1e6:.2f} million USD")

# -------------------------------------------------------
# 3. SENSITIVITY RANGES
# -------------------------------------------------------

percent_changes = np.array([-40, -30, -20, -10, 0, 10, 20, 30, 40])

def pct_to_factor(pct):
    return 1.0 + pct / 100.0

sensitivity_df = pd.DataFrame({"% change": percent_changes})

# -------------------------------------------------------
# 4. SENSITIVITY: PRICE OF RAW MATERIALS
# -------------------------------------------------------

NPV_raw = []
for pct in percent_changes:
    f = pct_to_factor(pct)
    raw_cost = rawmat_cost_base * f
    NPV = calc_NPV(FCI_base, Revenue_base, raw_cost,
                   opex_ex_raw_base, depreciation_base,
                   scrap_base, tax_rate, discount_rate, project_life)
    NPV_raw.append(NPV)

sensitivity_df["PriceRawMat_NPV"] = np.array(NPV_raw) / 1e6  # million USD

# -------------------------------------------------------
# 5. SENSITIVITY: REVENUE
# -------------------------------------------------------

NPV_revenue = []
for pct in percent_changes:
    f = pct_to_factor(pct)
    revenue = Revenue_base * f
    NPV = calc_NPV(FCI_base, revenue, rawmat_cost_base,
                   opex_ex_raw_base, depreciation_base,
                   scrap_base, tax_rate, discount_rate, project_life)
    NPV_revenue.append(NPV)

sensitivity_df["Revenue_NPV"] = np.array(NPV_revenue) / 1e6

# -------------------------------------------------------
# 6. SENSITIVITY: OPEX EXCLUDING RAW MATERIALS
# -------------------------------------------------------

NPV_opex_ex = []
for pct in percent_changes:
    f = pct_to_factor(pct)
    opex_ex_raw = opex_ex_raw_base * f
    NPV = calc_NPV(FCI_base, Revenue_base, rawmat_cost_base,
                   opex_ex_raw, depreciation_base,
                   scrap_base, tax_rate, discount_rate, project_life)
    NPV_opex_ex.append(NPV)

sensitivity_df["OPEX_exclRaw_NPV"] = np.array(NPV_opex_ex) / 1e6

# -------------------------------------------------------
# 7. SENSITIVITY: CAPEX (FCI)
# -------------------------------------------------------

NPV_capex = []
for pct in percent_changes:
    f = pct_to_factor(pct)
    FCI = FCI_base * f
    scrap = scrap_base * f   # scrap scales with FCI
    NPV = calc_NPV(FCI, Revenue_base, rawmat_cost_base,
                   opex_ex_raw_base, depreciation_base,
                   scrap, tax_rate, discount_rate, project_life)
    NPV_capex.append(NPV)

sensitivity_df["CAPEX_NPV"] = np.array(NPV_capex) / 1e6

print("\nNPV sensitivity table (million USD):")
print(sensitivity_df.to_string(index=False))

# -------------------------------------------------------
# 8. SPIDER DIAGRAM
# -------------------------------------------------------

plt.figure(figsize=(8, 5))

x = sensitivity_df["% change"].values

plt.plot(x, sensitivity_df["PriceRawMat_NPV"].values,
         marker='o', linewidth=2, label="Raw Materials Price")
plt.plot(x, sensitivity_df["Revenue_NPV"].values,
         marker='o', linewidth=2, label="Product Price")
plt.plot(x, sensitivity_df["OPEX_exclRaw_NPV"].values,
         marker='o', linewidth=2, label="OPEX excl. raw materials")
plt.plot(x, sensitivity_df["CAPEX_NPV"].values,
         marker='o', linewidth=2, label="CAPEX")

# plt.axvline(0, linewidth=1)
plt.axhline(NPV_base / 1e6, linestyle='--', linewidth=1)

plt.xlabel("Percentage change in variable (%)")
plt.ylabel("NPV [million USD]")
plt.title("NPV Sensitivity Spider Diagram")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
