from evaluate import *

# Define root path
root_path = 'Data/compare'
slew_dirs = ['slew1', 'slew2']


# Collect all data
summary = {}

for slew in slew_dirs:
    slew_path = os.path.join(root_path, slew)
    for k in os.listdir(slew_path):
        k_path = os.path.join(slew_path, k)
        if not os.path.isdir(k_path):
            continue
        records = []
        for filename in os.listdir(k_path):
            if filename.endswith('.mat'):
                file_path = os.path.join(k_path, filename)
                seed = os.path.splitext(filename)[0]
                null_norm, orth_norm = extract_components(file_path)
                record = {
                    'Slew Type': slew,
                    'k': int(k),
                    'Seed': int(seed),
                    'Zero Crossings': sum(count_zero_crossings(file_path)),
                    'Omega² Avg': sum(omega_squared_avg(file_path)),
                    'Energy': sum(energy(file_path)),
                    'Stiction Time (limit=100)': sum(time_stiction_accurate(file_path, limit=99.99)),
                    'Nullspace': null_norm,
                    'Orthogonal': orth_norm
                }
                records.append(record)
        summary[(slew, k)] = pd.DataFrame(records)

# Write to Excel
with pd.ExcelWriter('summary_metrics.xlsx') as writer:
    for (slew, k), df in summary.items():
        sheet_name = f"{slew}_k{k}"[:31]  # Excel sheet name limit
        df.to_excel(writer, sheet_name=sheet_name, index=False)

'Excel file summary_metrics.xlsx written with all metrics.'
