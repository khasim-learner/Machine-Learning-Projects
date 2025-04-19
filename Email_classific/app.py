from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from utils import mask_pii
from models import classify_email
import threading

app = FastAPI()

# Define the input data structure
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

# Gradio Interface Function
def classify_email_gradio(email):
    # Simulating the API call for Gradio Interface
    email_input = EmailInput(email=email)
    response = classify_email_route(email_input)
    return response

# Gradio Interface Setup
iface = gr.Interface(
    fn=classify_email_gradio,
    inputs=gr.Textbox(label="Enter Email"),
    outputs=["json", "json", "json", "json"],
    live=True,
)

def run_gradio():
    iface.launch(share=True)

# Run Gradio interface in a separate thread
gradio_thread = threading.Thread(target=run_gradio)
gradio_thread.start()

