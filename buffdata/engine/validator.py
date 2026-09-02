from typing import List, Tuple
from buffdata.models.schemas import DatasetItem
from buffdata.plugins import run_validator_plugins

class DatasetValidator:
    @staticmethod
    def validate_item(item: DatasetItem) -> List[str]:
        errors = []
        prompt, response = item.get_prompt_and_response()
        classifiable = item.get_classification_text()
        if not prompt and not response and not classifiable:
            errors.append("Missing usable text, instruction, prompt, or messages content.")
        if item.format.value == "alpaca" and item.instruction and not item.output:
            errors.append("Instruction format detected but output is missing.")
        if item.format.value == "chat" and not item.messages:
            errors.append("Chat format detected but messages is empty.")
        errors.extend(run_validator_plugins(item))
        return errors

    @staticmethod
    def validate(items: List[DatasetItem]) -> Tuple[bool, List[str]]:
        errors = []
        for i, item in enumerate(items):
            errors.extend(f"Row {i}: {error}" for error in DatasetValidator.validate_item(item))
        
        is_valid = len(errors) == 0
        return is_valid, errors
