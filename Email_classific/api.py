from fastapi import FastAPI
from models import classify_email
from utils import mask_pii
from pydantic import BaseModel

app = FastAPI()

class EmailInput(BaseModel):
    email: str

@app.post("/classify_email")
async def classify_email_route(email_input: EmailInput):
    email = email_input.email

    # Mask PII in the email
    masked_email, masked_entities = mask_pii(email)

    # Classify the email using the model
    category = classify_email(email)

    return {
        "input_email_body": email,
        "list_of_masked_entities": masked_entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
