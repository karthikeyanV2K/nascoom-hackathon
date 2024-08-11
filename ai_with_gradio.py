import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, SegformerForImageClassification, SegformerImageProcessor
from PIL import Image
import torch
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load models
text_model_name = "gpt2"  # or any other text model
text_model = AutoModelForCausalLM.from_pretrained(text_model_name)
tokenizer = AutoTokenizer.from_pretrained(text_model_name)

# Segformer model for image classification
seg_model_name = "nvidia/segformer-b3-finetuned-ade-512-512"
seg_model = SegformerForImageClassification.from_pretrained(seg_model_name)
seg_processor = SegformerImageProcessor.from_pretrained(seg_model_name)

# Load the dataset
soil_data = pd.read_csv("E:\\edii\\crop_data.csv")

# Separate features and target variable
X = soil_data.drop(columns=["Suitable_Seed"])
y = soil_data["Suitable_Seed"]

# Create a label encoder
label_encoder = LabelEncoder()

# Encode the target variable
y_encoded = label_encoder.fit_transform(y)

# Create a mapping of class labels to their encoded values
class_mapping = {label: i for i, label in enumerate(label_encoder.classes_)}

# Initialize the XGBoost classifier
model = XGBClassifier()

# Train the model
model.fit(X, y_encoded)


# Functions
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = text_model.generate(inputs["input_ids"], max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def classify_image(image):
    if image is None:
        return "No image provided."
    # Process the image directly
    inputs = seg_processor(images=image, return_tensors="pt")
    outputs = seg_model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()
    return f"Predicted class: {predicted_class_id}"


def predict_best_seed(soil_report):
    # Predict using the trained model
    predicted_label = model.predict([soil_report])[0]
    # Reverse the mapping to get the predicted seed
    predicted_seed = [key for key, value in class_mapping.items() if value == predicted_label][0]
    return predicted_seed


def process(text_input, image_input, pH, moisture, nitrogen, phosphorus, potassium):
    text_output = None
    image_output = None
    soil_output = None

    if text_input:
        # Only text provided
        text_output = generate_text(text_input)

    if image_input:
        # Only image provided
        image_output = classify_image(image_input)

    if pH is not None:
        # Soil report provided
        soil_report = [pH, moisture, nitrogen, phosphorus, potassium]
        soil_output = predict_best_seed(soil_report)

    return text_output, image_output, soil_output


# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        text_input = gr.Textbox(label="Enter your text")
        image_input = gr.Image(type="pil", label="Upload an image")

        # Soil attributes inputs
        pH_input = gr.Number(label="pH level", value=6.5, step=0.1)
        moisture_input = gr.Number(label="Moisture content (%)", value=20.0, step=0.1)
        nitrogen_input = gr.Number(label="Nitrogen level", value=50.0, step=1.0)
        phosphorus_input = gr.Number(label="Phosphorus level", value=30.0, step=1.0)
        potassium_input = gr.Number(label="Potassium level", value=40.0, step=1.0)

    output_text = gr.Textbox(label="Text Output")
    output_image = gr.Textbox(label="Image Classification Output")
    output_soil = gr.Textbox(label="Recommended Seed")

    submit_btn = gr.Button("Submit")
    submit_btn.click(process,
                     inputs=[text_input, image_input, pH_input, moisture_input, nitrogen_input, phosphorus_input,
                             potassium_input],
                     outputs=[output_text, output_image, output_soil])

# Launch with public link
demo.launch(share=True)
