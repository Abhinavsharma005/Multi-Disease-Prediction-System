import json
import sys

def inspect_notebook(filename, out_file):
    out_file.write("="*80 + "\n")
    out_file.write(f"INSPECTING: {filename}\n")
    out_file.write("="*80 + "\n")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        cells = nb.get('cells', [])
        code_idx = 1
        for cell in cells:
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    source_str = "".join(source)
                else:
                    source_str = source
                
                out_file.write(f"\n--- Code Cell {code_idx} ---\n")
                out_file.write(source_str.strip() + "\n")
                code_idx += 1
    except Exception as e:
        out_file.write(f"Error reading {filename}: {e}\n")

if __name__ == '__main__':
    notebooks = [
        'symptom-based-disease-prediction.ipynb',
        'HeartdiseaseFinal.ipynb',
        'diabetes-prediction.ipynb',
        'mental-health-risk-prediction.ipynb'
    ]
    with open('notebooks_inspect_output.txt', 'w', encoding='utf-8') as out_file:
        for nb in notebooks:
            inspect_notebook(nb, out_file)
    print("Done!")
