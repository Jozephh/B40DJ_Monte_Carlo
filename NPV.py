import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy_financial as nf
import time

project_life = 30
tax_rate_base = 0.15
discount_rate_base = 0.07

# -------------------------------------------------------
# BASE-CASE VALUES
# -------------------------------------------------------

# CAPEX / FCI
FCI_base = 75_331_279

# Annual revenue (base case)
Revenue_base = 247_199_320

# Scrap value = 10% of CAPEX
scrap_base = 0.10 * FCI_base

# Annual cost of raw materials at base case (USD/year)
rawmat_cost_base = 186_893_776

# Annual OPEX excluding raw materials (USD/year)
opex_ex_raw_base = 32_030_755

# CAPEX split assumptions
capex_split_year1 = 0.40
capex_split_year2 = 0.60

# First production year assumptions
startup_factor = 0.60

# -------------------------------------------------------
# DEPRECIATION RATE FROM FORMULA
# -------------------------------------------------------
# Depreciation rate = 1 - (Scrap Value / Initial Capital)^(1/N)

depreciation_rate = 1 - (scrap_base / FCI_base) ** (1 / project_life)

print(f"Calculated depreciation rate = {depreciation_rate * 100:.4f}% per year")

# -------------------------------------------------------
# CASH FLOW MODEL
# -------------------------------------------------------

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
    """
    Timeline:
    Year 0: no cash flow
    Year 1: 40% CAPEX
    Year 2: 60% CAPEX, no production yet
    Year 3: first production year at startup_factor capacity and OPEX
    Year 4..(N-1): full operation
    Year N: full operation + scrap

    Depreciation method:
    Constant-rate declining-balance depreciation using:
        rate = 1 - (scrap / FCI)^(1/N)

    Book value declines each operating year and should reach scrap value
    at the end of the project life.
    """

    CF = np.zeros(project_life + 1)
    depreciation_schedule = np.zeros(project_life + 1)
    book_values = np.zeros(project_life + 1)

    # Initial book value before operation starts
    book_value = FCI

    # CAPEX split over Year 1 and Year 2
    CF[1] = -capex_split_year1 * FCI
    CF[2] = -capex_split_year2 * FCI

    # Number of operating years = Year 3 to Year project_life inclusive
    operating_years = project_life - 2

    # Build depreciation schedule from Year 3 to Year N
    for i, t in enumerate(range(3, project_life + 1), start=1):
        remaining_years = operating_years - i + 1

        if t < project_life:
            depreciation = depreciation_rate * book_value
            # Prevent book value dropping below scrap too early
            if book_value - depreciation < scrap:
                depreciation = book_value - scrap
        else:
            # Final year: force ending book value exactly to scrap
            depreciation = book_value - scrap

        depreciation_schedule[t] = depreciation
        book_value -= depreciation
        book_values[t] = book_value

    # Year 3: startup year
    startup_revenue = startup_factor * revenue
    startup_raw = startup_factor * raw_cost
    startup_opex_ex = startup_factor * opex_ex_raw
    depr_startup = depreciation_schedule[3]

    taxable_income_startup = startup_revenue - startup_raw - startup_opex_ex - depr_startup
    tax_startup = tax_rate * taxable_income_startup if taxable_income_startup > 0 else 0.0
    CF[3] = taxable_income_startup - tax_startup + depr_startup

    # Years 4 to project_life-1: full operation
    total_opex = raw_cost + opex_ex_raw

    for t in range(4, project_life):
        depr = depreciation_schedule[t]
        taxable_income = revenue - total_opex - depr
        tax = tax_rate * taxable_income if taxable_income > 0 else 0.0
        CF[t] = taxable_income - tax + depr

    # Final year: full operation + scrap
    depr_final = depreciation_schedule[project_life]
    taxable_income_final = revenue - total_opex - depr_final + scrap
    tax_final = tax_rate * taxable_income_final if taxable_income_final > 0 else 0.0
    CF[project_life] = taxable_income_final - tax_final + depr_final

    return CF, depreciation_schedule, book_values


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


# -------------------------------------------------------
# BASE-CASE CF AND NPV / IRR
# -------------------------------------------------------

years = np.arange(project_life + 1)

CF_base, depreciation_schedule_base, book_values_base = build_after_tax_CF(
    FCI=FCI_base,
    revenue=Revenue_base,
    raw_cost=rawmat_cost_base,
    opex_ex_raw=opex_ex_raw_base,
    depreciation_rate=depreciation_rate,
    scrap=scrap_base,
    tax_rate=tax_rate_base,
    project_life=project_life,
    startup_factor=startup_factor
)

NPV_base = np.sum(CF_base / (1 + discount_rate_base) ** years)
IRR_base = nf.irr(CF_base)
base_PBT = calc_payback_time(CF_base)

# -------------------------------------------------------
# OUTPUTS
# -------------------------------------------------------

print(f"\nDepreciation rate used      = {depreciation_rate * 100:.4f}% per year")
print(f"Initial CAPEX              = {FCI_base:,.2f} USD")
print(f"Scrap value (10% CAPEX)    = {scrap_base:,.2f} USD")
print(f"Final book value           = {book_values_base[project_life]:,.2f} USD")

print(f"\nBase-case NPV              = {NPV_base/1e6:.2f} million USD")
print(f"Base-case IRR              = {IRR_base*100:.2f} %")
print(f"Base-case Payback Time     = {base_PBT:.2f} years")

# Optional table
# -------------------------------------------------------
# BUILD FULL FINANCIAL TABLE
# -------------------------------------------------------

years = np.arange(project_life + 1)

investments = np.zeros(project_life + 1)
revenue_array = np.zeros(project_life + 1)
costs_array = np.zeros(project_life + 1)
taxes_array = np.zeros(project_life + 1)

# CAPEX
investments[1] = capex_split_year1 * FCI_base
investments[2] = capex_split_year2 * FCI_base

# Costs
total_opex = rawmat_cost_base + opex_ex_raw_base

for t in range(3, project_life + 1):

    if t == 3:
        revenue_array[t] = startup_factor * Revenue_base
        costs_array[t] = startup_factor * total_opex
    else:
        revenue_array[t] = Revenue_base
        costs_array[t] = total_opex

    # depreciation
    depr = depreciation_schedule_base[t]

    taxable_income = revenue_array[t] - costs_array[t] - depr

    if t == project_life:
        taxable_income += scrap_base

    tax = tax_rate_base * taxable_income if taxable_income > 0 else 0
    taxes_array[t] = tax

# Annual NCF already calculated
annual_ncf = CF_base
cumulative_ncf = np.cumsum(annual_ncf)

# -------------------------------------------------------
# CREATE TABLE
# -------------------------------------------------------

table = pd.DataFrame({
    "Year": years,
    "Investments ($)": investments,
    "Revenue ($)": revenue_array,
    "Costs ($)": costs_array,
    "Taxes ($)": taxes_array,
    "Annual NCF ($)": annual_ncf,
    "Cumulative NCF ($)": cumulative_ncf
})

# -------------------------------------------------------
# ADD TOTAL ROW
# -------------------------------------------------------

totals = pd.DataFrame({
    "Year": ["Total"],
    "Investments ($)": [investments.sum()],
    "Revenue ($)": [revenue_array.sum()],
    "Costs ($)": [costs_array.sum()],
    "Taxes ($)": [taxes_array.sum()],
    "Annual NCF ($)": [""],
    "Cumulative NCF ($)": [""]
})

table = pd.concat([table, totals], ignore_index=True)

# -------------------------------------------------------
# PRINT FULL TABLE
# -------------------------------------------------------

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:,.2f}'.format)

print("\nFull Financial Table:")
print(table.to_string(index=False))

print("\nCash flow summary:")
print(table.round(2))