import scipy.io
import os
import pandas as pd
import numpy as np

def calculate_metric(w_sol):
    if w_sol is None:
        return None
    return np.mean(np.sum(w_sol**2, axis=0))

# Update this path to your folder of .mat files
mat_folder = "slew1"
output_excel = "slew1/slew1_results.xlsx"

# Class names (in order)
expected_class_names = ["ref", "osl", "tsl", "osl_tsl", "og", "cg", "og_cg"]

def load_classes_from_mat(filename):
    mat_data = scipy.io.loadmat(filename, squeeze_me=True, struct_as_record=False)

    class_list = []
    for key, value in mat_data.items():
        if key.startswith('__'):
            continue

        class_data = {}
        if hasattr(value, '_fieldnames'):
            for field in value._fieldnames:
                attr_value = getattr(value, field)
                class_data[field] = attr_value
            class_list.append(class_data)
        else:
            print(f"Warning: {key} is not a struct-like entry.")
    return class_list

# Collect data for all files
all_data = []

for file in os.listdir(mat_folder):
    if file.endswith(".mat"):
        full_path = os.path.join(mat_folder, file)
        loaded_classes = load_classes_from_mat(full_path)

        if len(loaded_classes) != len(expected_class_names):
            print(f"Warning: {file} has {len(loaded_classes)} classes instead of 7")
            continue

        for class_data, class_name in zip(loaded_classes, expected_class_names):
            solve_time = class_data.get('solve_time', None)
            setup_time = class_data.get('setup_time', None)
            iteration_count = class_data.get('iteration_count', None)
            w_sol = class_data.get('w_sol', None)
            metric = calculate_metric(w_sol)

            all_data.append({
                "file": full_path,
                "class": class_name,
                "solve_time": solve_time,
                "setup_time": setup_time,
                "iteration_count": iteration_count,
                "metric": metric
            })

# Convert to DataFrame and write to Excel
df = pd.DataFrame(all_data)
df.to_excel(output_excel, index=False)

print(f"Saved extracted data to {output_excel}")
