import re

def mask_pii(text):
    entities = []
    entity_patterns = {
        "full_name": r"(?i)(?<=name is )([A-Z][a-z]+\s[A-Z][a-z]+)",
        "email": r"[\w\.-]+@[\w\.-]+",
        "phone_number": r"\b\d{10}\b",
        "dob": r"\b\d{2}/\d{2}/\d{4}\b",
        "aadhar_num": r"\b\d{4} \d{4} \d{4}\b",
        "credit_debit_no": r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",
        "cvv_no": r"\b\d{3}\b",
        "expiry_no": r"\b\d{2}/\d{2}\b"
    }

    for label, pattern in entity_patterns.items():
        for match in re.finditer(pattern, text):
            entity_text = match.group()
            start, end = match.span()
            entities.append({
                "position": [start, end],
                "classification": label,
                "entity": entity_text
            })
            text = text[:start] + f"[{label}]" + text[end:]

    return text, entities
