import pandas as pd
import numpy as np

print("a")
filepath = 'C:/Users/Krzysiek/Downloads/dane_overworld.xlsx'
xls = pd.ExcelFile(filepath)
print("Sheet names:", xls.sheet_names)

df = pd.read_excel(filepath, sheet_name=xls.sheet_names[0])
print("\nShape:", df.shape)
print(df.info())
print("\nColumns:", df.columns.tolist())
print("\nInfo:")
print(df.head(10))
print("\nHead:")

print("Unique status values and counts:")
print(df['status'].value_counts())

print("\nNumber of unique biomes:", df['biome'].nunique())
print("Biome counts (top 15 and bottom 15):")
biome_counts = df['biome'].value_counts()
print(biome_counts)

print("\nCheck dx = end_x - start_x:")
df['dx'] = df['end_x'] - df['start_x']
print(df['dx'].describe())

print("\nCheck speed_bps vs dx / time_sec:")
df['calc_speed_x'] = df['dx'] / df['time_sec']
df['calc_speed_3d'] = df['total_distance_3d'] / df['time_sec']
print((df['speed_bps'] - df['calc_speed_x']).abs().max(), (df['speed_bps'] - df['calc_speed_3d']).abs().max())

# Let's inspect TIMEOUT rows and negative dx rows
print("TIMEOUT rows summary:")
timeouts = df[df['status'] == 'TIMEOUT']
print(timeouts[['biome', 'dx', 'time_sec', 'speed_bps']].head(15))
print("\nTIMEOUT counts per biome:")
print(timeouts['biome'].value_counts())

print("\nAny SUCCESS rows with weird dx or speed_bps <= 0?")
succ = df[df['status'] == 'SUCCESS'].copy()
print("Min dx in SUCCESS:", succ['dx'].min(), "Max dx in SUCCESS:", succ['dx'].max())
print("Min speed_bps in SUCCESS:", succ['speed_bps'].min(), "Max speed_bps in SUCCESS:", succ['speed_bps'].max())
print("Min time_sec in SUCCESS:", succ['time_sec'].min(), "Max time_sec in SUCCESS:", succ['time_sec'].max())

# Check how speed_bps is calculated on SUCCESS rows
print("\nMax diff between speed_bps and dx/time_sec on SUCCESS:")
print((succ['speed_bps'] - (succ['dx'] / succ['time_sec'])).abs().max())

# Inspect high speed_bps rows (e.g., > 7 bps) to see what happened
print("Top 10 fastest rows:")
print(succ.sort_values(by='speed_bps', ascending=False)[['biome', 'dx', 'time_sec', 'speed_bps', 'total_distance_3d', 'water_ticks', 'elevation_gain', 'elevation_loss']].head(10))

# Calculate cost_sec_per_block and cost_sec_per_chunk
succ['sec_per_block'] = succ['time_sec'] / succ['dx']
succ['sec_per_chunk'] = succ['sec_per_block'] * 16.0

# Also check what happens if we penalize TIMEOUTs (e.g. assigning them a high cost like 60s for the fraction or capping) vs filtering them out + adding a timeout penalty rate
df['timeout_flag'] = (df['status'] == 'TIMEOUT').astype(int)

# Group by biome on SUCCESS rows
stats = succ.groupby('biome').agg(
    n_success=('sec_per_chunk', 'count'),
    median_sec_chunk=('sec_per_chunk', 'median'),
    mean_sec_chunk=('sec_per_chunk', 'mean'),
    std_sec_chunk=('sec_per_chunk', 'std'),
    iqr_sec_chunk=('sec_per_chunk', lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
    mad_sec_chunk=('sec_per_chunk', lambda x: np.median(np.abs(x - np.median(x))) * 1.4826), # normalized MAD
    p90_sec_chunk=('sec_per_chunk', lambda x: np.percentile(x, 90)),
    mean_speed_bps=('speed_bps', 'mean'),
    median_speed_bps=('speed_bps', 'median'),
    mean_water_ticks=('water_ticks', 'mean'),
    mean_elev_gain=('elevation_gain', 'mean')
).reset_index()

# Add total attempts and timeout rate
total_counts = df.groupby('biome').agg(
    n_total=('status', 'count'),
    n_timeout=('timeout_flag', 'sum')
).reset_index()

stats = pd.merge(stats, total_counts, on='biome')
stats['timeout_rate'] = stats['n_timeout'] / stats['n_total']

# Risk-adjusted cost formulas:
# 1. Parametric: mean + 0.5 * std
# 2. Robust (recommended due to skewness/outliers): median + 0.5 * mad (or IQR/2) + timeout_penalty
# Let's see how both look!
lambda_risk = 0.5
# If a bot times out (60s limit without reaching 100 blocks), it represents severe pathfinding failure / impassable cliff.
# Adding a penalty proportional to timeout_rate (e.g., + 10.0 * timeout_rate sec/chunk) penalizes risky biomes like shattered_savanna_plateau
stats['risk_cost_robust'] = stats['median_sec_chunk'] + lambda_risk * stats['mad_sec_chunk'] + (10.0 * stats['timeout_rate'])
stats['risk_cost_param'] = stats['mean_sec_chunk'] + lambda_risk * stats['std_sec_chunk'] + (10.0 * stats['timeout_rate'])

stats_sorted = stats.sort_values(by='risk_cost_robust')

pd.set_option('display.max_rows', 60)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
print(stats_sorted[['biome', 'n_total', 'n_timeout', 'timeout_rate', 'median_sec_chunk', 'mean_sec_chunk', 'std_sec_chunk', 'mad_sec_chunk', 'risk_cost_robust', 'risk_cost_param']].round(3))

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), dpi=150)

# Top 15 fastest biomes
top15 = stats_sorted.head(15).sort_values(by='risk_cost_robust', ascending=True)
y_pos1 = np.arange(len(top15))
ax1.barh(y_pos1, top15['median_sec_chunk'], color='#2b8cbe', label='Mediana kosztu bazowego [s/chunk]')
ax1.barh(y_pos1, top15['risk_cost_robust'] - top15['median_sec_chunk'], left=top15['median_sec_chunk'], color='#f03b20', alpha=0.8, label='Narzut ryzyka (0.5*MAD + kara TIMEOUT)')
ax1.set_yticks(y_pos1)
ax1.set_yticklabels(top15['biome'], fontsize=10)
ax1.invert_yaxis()
ax1.set_xlabel('Koszt przejścia [sekundy / chunk 16 bloków]', fontsize=10)
ax1.set_title('TOP 15 najszybszych i najbardziej stabilnych biomów', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 7.2)
ax1.legend(loc='lower right', fontsize=9)

for i, v in enumerate(top15['risk_cost_robust']):
    ax1.text(v + 0.08, i, f"{v:.2f}s", va='center', fontsize=9)

# Top 15 slowest / riskiest biomes
bot15 = stats_sorted.tail(15).sort_values(by='risk_cost_robust', ascending=True)
y_pos2 = np.arange(len(bot15))
ax2.barh(y_pos2, bot15['median_sec_chunk'], color='#2b8cbe', label='Mediana kosztu bazowego [s/chunk]')
ax2.barh(y_pos2, bot15['risk_cost_robust'] - bot15['median_sec_chunk'], left=bot15['median_sec_chunk'], color='#f03b20', alpha=0.8, label='Narzut ryzyka (0.5*MAD + kara TIMEOUT)')
ax2.set_yticks(y_pos2)
ax2.set_yticklabels(bot15['biome'], fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel('Koszt przejścia [sekundy / chunk 16 bloków]', fontsize=10)
ax2.set_title('TOP 15 najwolniejszych i najbardziej ryzykownych biomów', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 7.2)
ax2.legend(loc='lower right', fontsize=9)

for i, v in enumerate(bot15['risk_cost_robust']):
    ax2.text(v + 0.08, i, f"{v:.2f}s", va='center', fontsize=9)

plt.tight_layout()
plt.savefig('biome_costs_comparison.png', bbox_inches='tight')
print("Saved chart successfully.")

# Export clean lookup table to CSV for user's A* algorithm
export_cols = ['biome', 'n_total', 'n_timeout', 'timeout_rate', 'median_sec_chunk', 'mean_sec_chunk', 'std_sec_chunk', 'mad_sec_chunk', 'risk_cost_robust', 'risk_cost_param']
stats_sorted[export_cols].round(4).to_csv('biome_cost_lookup.csv', index=False)
