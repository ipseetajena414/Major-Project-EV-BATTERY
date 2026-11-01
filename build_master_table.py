import pandas as pd

# 1. Load the raw datasets
df_nmc_lfp = pd.read_csv('ev battery_failure prediction Dataset.csv')
df_sodium = pd.read_csv('synthetic_battery_lifecycle.csv')
df_solid = pd.read_csv('solid_state_ev_battery_telemetry_enriched_2026.csv')

# 2. Extract and Map (NMC & LFP)
df_nmc_lfp = df_nmc_lfp[df_nmc_lfp['battery_chemistry'].isin(['NMC', 'LFP'])].copy()
df_1 = pd.DataFrame({
    'battery_chemistry': df_nmc_lfp['battery_chemistry'],
    'cycle_id': df_nmc_lfp['cycle_count'],
    'time_sec': df_nmc_lfp['average_charging_time'] * 60,
    'voltage_v': df_nmc_lfp['cell_voltage_avg'],
    'current_a': (df_nmc_lfp['average_charge_power_kw'] * 1000) / df_nmc_lfp['pack_voltage'],
    'temp_c': df_nmc_lfp['cell_temperature_avg'],
    'capacity_ah': df_nmc_lfp['remaining_capacity'],
    'soh_pct': df_nmc_lfp['battery_health_percent']
})

# 3. Extract and Map (Na-ion)
df_2 = pd.DataFrame({
    'battery_chemistry': 'Na-ion',
    'cycle_id': df_sodium['cycle_count'],
    'time_sec': df_sodium['charge_time_mins'] * 60,
    'voltage_v': 3.1, 
    'current_a': 15.0, 
    'temp_c': df_sodium['temperature_celsius'],
    'capacity_ah': df_sodium['discharge_capacity_ah'],
    'soh_pct': 100 - (df_sodium['internal_resistance_mohms'] / 100)
})

# 4. Extract and Map (Solid-State)
df_3 = pd.DataFrame({
    'battery_chemistry': 'Solid-State',
    'cycle_id': df_solid['cycle_number'],
    'time_sec': (df_solid['nominal_capacity_ah'] / df_solid['charging_c_rate']) * 3600, 
    'voltage_v': df_solid['nominal_voltage_plateau_v'],
    'current_a': df_solid['nominal_capacity_ah'] * df_solid['charging_c_rate'], 
    'temp_c': df_solid['operating_temperature_c'],
    'capacity_ah': df_solid['nominal_capacity_ah'],
    'soh_pct': df_solid['state_of_health_pct']
})

# 5. Fuse and Apply Blueprint Constraints
master_df = pd.concat([df_1, df_2, df_3], ignore_index=True)
master_df = master_df.dropna(subset=['cycle_id', 'soh_pct', 'voltage_v'])

# CONSTRAINT 1: Enforce strict sensor noise clipping between 2.0V and 4.5V
master_df['voltage_v'] = master_df['voltage_v'].clip(lower=2.0, upper=4.5)

# CONSTRAINT 2: Bound SOH between realistic percentages (0 to 100)
master_df['soh_pct'] = master_df['soh_pct'].clip(lower=0.0, upper=100.0)

# CONSTRAINT 3: Sort sequentially for Member 2's LSTM sliding window ingestion
master_df = master_df.sort_values(by=['battery_chemistry', 'cycle_id']).reset_index(drop=True)

# 6. Type Enforcement and Export
master_df['cycle_id'] = master_df['cycle_id'].astype(int)
master_df['time_sec'] = master_df['time_sec'].astype(float).round(1)
master_df['voltage_v'] = master_df['voltage_v'].astype(float).round(3)
master_df['current_a'] = master_df['current_a'].astype(float).round(2)
master_df['temp_c'] = master_df['temp_c'].astype(float).round(2)
master_df['capacity_ah'] = master_df['capacity_ah'].astype(float).round(2)
master_df['soh_pct'] = master_df['soh_pct'].astype(float).round(2)

master_df.to_parquet('master_table.parquet', index=False)
print("Perfected master table saved with blueprint constraints applied.")