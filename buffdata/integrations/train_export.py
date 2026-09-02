import json
import polars as pl

def export_unsloth_sharegpt(input_jsonl: str, output_json: str):
    # Unsloth prefers ShareGPT format:
    # [{"conversations": [{"from": "human", "value": "instruction"}, {"from": "gpt", "value": "output"}]}]
    df = pl.read_ndjson(input_jsonl)
    
    formatted = []
    for row in df.iter_rows(named=True):
        instruction = row.get('instruction', '')
        output = row.get('output', '')
        if not instruction or not output:
            continue
            
        formatted.append({
            "conversations": [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        })
        
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(formatted, f, indent=2)
    print(f"Exported {len(formatted)} items to {output_json} in ShareGPT format suitable for Unsloth.")
