import math
from typing import List, Tuple
from buffdata.models.schemas import DatasetItem

class PerplexityFilter:
    def __init__(self, model_id: str = "gpt2"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def calculate_perplexity(self, text: str) -> float:
        import torch
        if not text.strip():
            return float('inf')
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        max_length = self.model.config.n_positions
        
        # If text is too long, truncate it
        if inputs.input_ids.size(1) > max_length:
            inputs.input_ids = inputs.input_ids[:, :max_length]
            if "attention_mask" in inputs:
                inputs.attention_mask = inputs.attention_mask[:, :max_length]
                
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, labels=inputs.input_ids)
            loss = outputs.loss
            
        return math.exp(loss.item())

    def filter_batch(self, items: List[DatasetItem], max_ppl: float) -> Tuple[List[DatasetItem], List[DatasetItem]]:
        kept = []
        dropped = []
        
        for item in items:
            d = item.to_dict()
            text = d.get("output") or d.get("text") or d.get("instruction") or ""
            ppl = self.calculate_perplexity(text)
            
            item.metadata = item.metadata or {}
            item.metadata["perplexity"] = ppl
            
            if ppl <= max_ppl:
                kept.append(item)
            else:
                dropped.append(item)
                
        return kept, dropped
