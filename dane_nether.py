import pandas as pd
import numpy as np

filepath = 'C:/Users/Krzysiek/Downloads/dane_nether.xlsx'
xls = pd.ExcelFile(filepath)
print("Sheet names:", xls.sheet_names)

for sheet in xls.sheet_names:
    df_tmp = pd.read_excel(filepath, sheet_name=sheet)
    print(f"\n--- Sheet: {sheet} | Shape: {df_tmp.shape} ---")
    print("Columns:", df_tmp.columns.tolist())
    print(df_tmp.head(5))

df = pd.read_excel('C:/Users/Krzysiek/Downloads/dane_nether.xlsx', sheet_name='Arkusz1')

print("Status value counts:")
print(df['status'].value_counts())

print("\nBiome value counts:")
print(df['biome'].value_counts())

df['dx'] = df['end_x'] - df['start_x']
print("\ndx summary on all rows:")
print(df['dx'].describe())

print("\nTIMEOUT rows breakdown:")
timeouts = df[df['status'] != 'SUCCESS']
print(timeouts.groupby(['biome', 'status']).size())
print("\nSample non-SUCCESS rows:")
print(timeouts[['status', 'biome', 'dx', 'time_sec', 'speed_bps']].head(15))

succ = df[df['status'] == 'SUCCESS'].copy()
print("\nSUCCESS dx min/max:", succ['dx'].min(), succ['dx'].max())
print("SUCCESS speed_bps min/max:", succ['speed_bps'].min(), succ['speed_bps'].max())
print("SUCCESS time_sec min/max:", succ['time_sec'].min(), succ['time_sec'].max())

print("\nTop 10 fastest rows in SUCCESS:")
print(succ.sort_values(by='speed_bps', ascending=False)[['biome', 'dx', 'time_sec', 'speed_bps', 'total_distance_3d', 'elevation_gain', 'elevation_loss']].head(10))

print("\nTop 10 slowest rows in SUCCESS:")
print(succ.sort_values(by='speed_bps', ascending=True)[['biome', 'dx', 'time_sec', 'speed_bps', 'total_distance_3d', 'elevation_gain', 'elevation_loss']].head(10))
succ['sec_per_block'] = succ['time_sec'] / succ['dx']
succ['sec_per_chunk'] = succ['sec_per_block'] * 16.0
df['timeout_flag'] = (df['status'] == 'TIMEOUT').astype(int)

stats = succ.groupby('biome').agg(
    n_success=('sec_per_chunk', 'count'),
    median_sec_chunk=('sec_per_chunk', 'median'),
    mean_sec_chunk=('sec_per_chunk', 'mean'),
    std_sec_chunk=('sec_per_chunk', 'std'),
    mad_sec_chunk=('sec_per_chunk', lambda x: np.median(np.abs(x - np.median(x))) * 1.4826),
    min_sec_chunk=('sec_per_chunk', 'min'),
    max_sec_chunk=('sec_per_chunk', 'max'),
    median_speed_bps=('speed_bps', 'median'),
    mean_elev_gain=('elevation_gain', 'mean')
).reset_index()

total_counts = df.groupby('biome').agg(
    n_total=('status', 'count'),
    n_timeout=('timeout_flag', 'sum')
).reset_index()

stats = pd.merge(stats, total_counts, on='biome')
stats['timeout_rate'] = stats['n_timeout'] / stats['n_total']

lambda_risk = 0.5
stats['risk_cost_robust'] = stats['median_sec_chunk'] + lambda_risk * stats['mad_sec_chunk'] + (10.0 * stats['timeout_rate'])
stats['risk_cost_param'] = stats['mean_sec_chunk'] + lambda_risk * stats['std_sec_chunk'] + (10.0 * stats['timeout_rate'])

stats_sorted = stats.sort_values(by='risk_cost_robust').reset_index(drop=True)

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
print(stats_sorted[['biome', 'n_total', 'n_timeout', 'timeout_rate', 'median_sec_chunk', 'mean_sec_chunk', 'std_sec_chunk', 'mad_sec_chunk', 'risk_cost_robust', 'risk_cost_param', 'median_speed_bps', 'mean_elev_gain']].round(4))

export_cols = ['biome', 'n_total', 'n_timeout', 'timeout_rate', 'median_sec_chunk', 'mean_sec_chunk', 'std_sec_chunk', 'mad_sec_chunk', 'risk_cost_robust', 'risk_cost_param']
stats_sorted[export_cols].round(4).to_csv('nether_biome_cost_lookup.csv', index=False)
print("\nSaved nether_biome_cost_lookup.csv successfully.")
