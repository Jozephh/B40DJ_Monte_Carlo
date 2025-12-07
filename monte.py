import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl

data0 = pd.read_excel(r"C:\Users\joe\OneDrive - Heriot-Watt University\Uni_Cloud\Year 4\B40DJ_Sus_Mgmt_&_Process_Econ\Assignment 2\Assignment 2.xlsm", sheet_name=0)
data1 = pd.read_excel(r"C:\Users\joe\OneDrive - Heriot-Watt University\Uni_Cloud\Year 4\B40DJ_Sus_Mgmt_&_Process_Econ\Assignment 2\Assignment 2.xlsm", sheet_name=1)
data2 = pd.read_excel(r"C:\Users\joe\OneDrive - Heriot-Watt University\Uni_Cloud\Year 4\B40DJ_Sus_Mgmt_&_Process_Econ\Assignment 2\Assignment 2.xlsm", sheet_name=2)
data3 = pd.read_excel(r"C:\Users\joe\OneDrive - Heriot-Watt University\Uni_Cloud\Year 4\B40DJ_Sus_Mgmt_&_Process_Econ\Assignment 2\Assignment 2.xlsm", sheet_name=3)
data4 = pd.read_excel(r"C:\Users\joe\OneDrive - Heriot-Watt University\Uni_Cloud\Year 4\B40DJ_Sus_Mgmt_&_Process_Econ\Assignment 2\Assignment 2.xlsm", sheet_name=4)

# -----------------------------
# 1. BASE CASE
# -----------------------------

N_SIM = 100000

# 20 year plant life
project_life = 20

# pull discount rate from excel sheet
discount_rate = data4.iloc[1,13]
discount_rate = int(discount_rate)

FCI_base   = data4.iloc[2,4]   
FCI_base = int(FCI_base)       # from Year 0

Revenues_base = data4.iloc[17,3]   
Revenues_base = int(Revenues_base)   # from table

OPEX_base  = data4.iloc[18,3]        # from table
OPEX_base = int(OPEX_base)

Sigma_Ce = data4.iloc[2,3]           # from table
Sigma_Ce = int(Sigma_Ce)

plant_cap_base = data3.iloc[0,2]    # plant capacity factor 
plant_cap_base = int(plant_cap_base)

scrap_base = 0.10 * Sigma_Ce     # 10 % of equipment cost

# -----------------------------
# 2. UNCERTAINTY RANGES
# -----------------------------
rev_low, rev_high   = 0.6, 1.4   # ±40% on revenue
opex_low, opex_high = 0.6, 1.4  # ±40% on OPEX
capex_low, capex_high = 0.95, 1.05 # ±5% on FCI and scrap
plant_cap_low, plant_cap_high = 0.8, 1 # -20% on plant capacity could be due to maintenance, emergency shutdown etc

# Sample random multipliers
rev_factor   = np.random.uniform(rev_low, rev_high,   N_SIM)
opex_factor  = np.random.uniform(opex_low, opex_high, N_SIM)
capex_factor = np.random.uniform(capex_low, capex_high, N_SIM)
plant_cap_factor = np.random.uniform(plant_cap_low, plant_cap_high, N_SIM)

NPV = np.zeros(N_SIM)

# -----------------------------
# 3. MONTE CARLO LOOP
# -----------------------------
for i in range(N_SIM):

    R    = Revenues_base * rev_factor[i]
    OPEX = OPEX_base * opex_factor[i]
    FCI  = FCI_base * capex_factor[i]
    scrap = scrap_base * capex_factor[i]
    pcap = plant_cap_base * plant_cap_factor[i]

    CF = np.zeros(project_life + 1)
    CF[0] = -FCI

    for year in range(1, project_life + 1):
        CF[year] = R - OPEX   # no depreciation, no tax

    CF[-1] += scrap   # add scrap value in final year

    years = np.arange(project_life + 1)
    NPV[i] = np.sum(CF / (1 + discount_rate) ** years)

# -----------------------------
# 4. SUMMARY STATS
# -----------------------------
mean_NPV = np.mean(NPV)
p5 = np.percentile(NPV, 5)
p95 = np.percentile(NPV, 95)
prob_positive = np.mean(NPV > 0)
corr_rev   = np.corrcoef(rev_factor, NPV)[0,1]
corr_opex  = np.corrcoef(opex_factor, NPV)[0,1]
corr_capex = np.corrcoef(capex_factor, NPV)[0,1]
corr_pcap = np.corrcoef(plant_cap_factor, NPV)[0,1]

print(f"Mean NPV      = ${mean_NPV/1e6:.2f} M")
print(f"5th percentile = ${p5/1e6:.2f} M")
print(f"95th percentile= ${p95/1e6:.2f} M")
print(f"P(NPV > 0) = {prob_positive*100:.1f} %")
print("Correlation with NPV:")
print(f"Revenue factor: {corr_rev:.3f}")
print(f"OPEX factor: {corr_opex:.3f}")
print(f"CAPEX factor: {corr_capex:.3f}")
print(f"pcap factor: {corr_pcap:.3f}")

# -----------------------------
# 5. PLOTS
# -----------------------------
NPV_M = NPV / 1e6

# (a) Histogram
plt.figure(figsize=(8,5))
plt.hist(NPV_M, bins=40, edgecolor='black', alpha=0.7)
plt.axvline(mean_NPV/1e6, color='red', linestyle='--',
            label=f"Mean = {mean_NPV/1e6:.1f} M")
plt.axvline(p5/1e6,  color='green', linestyle=':',
            label=f"5th perc = {p5/1e6:.1f} M")
plt.axvline(p95/1e6, color='orange', linestyle=':',
            label=f"95th perc = {p95/1e6:.1f} M")
plt.xlabel("NPV [million USD]")
plt.ylabel("Frequency")
plt.title("Monte Carlo NPV Distribution")
plt.legend()
plt.tight_layout()

# (b) CDF / "all data" view
NPV_sorted = np.sort(NPV_M)
cum_prob = np.linspace(0, 1, N_SIM)

plt.figure(figsize=(8,5))
plt.plot(NPV_sorted, cum_prob, linewidth=2)
plt.axvline(0, color='grey', linestyle='--', label="NPV = 0")
plt.xlabel("NPV [million USD]")
plt.ylabel("Cumulative probability")
plt.title("Cumulative Distribution of NPV (Monte Carlo)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
