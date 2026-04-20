import json
import sys

path = '15_Interactive_Plotting_Monte_Carlo_Simulations.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if 'c_1, c_2, c_3 = plt.subplots(1, 3, figsize=(18, 6))' in line:
                source[i] = line.replace('c_1, c_2, c_3 = plt.subplots', 'fig, (c_1, c_2, c_3) = plt.subplots')
                print("Fixed subplots line.")
            if 'import pandas as pd' in line and not any('import matplotlib.pyplot' in l for l in source):
                source.insert(i + 1, 'import matplotlib.pyplot as plt\n')
                print("Added matplotlib import.")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)

print("Saved notebook.")
