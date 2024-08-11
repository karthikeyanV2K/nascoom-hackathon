import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Ollama 3 model and tokenizer
model_name = "huggingface/ollama-3"  # Replace with the actual model name if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    # Generate a response from the model
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=150)  # Adjust max_length as needed
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def agriculture_assistant(query):
    # Process query and return response
    return generate_response(query)

# Create a Gradio interface
iface = gr.Interface(
    fn=agriculture_assistant,
    inputs=gr.inputs.Textbox(label="Ask your agriculture question"),
    outputs=gr.outputs.Textbox(label="Response"),
    title="Agriculture Assistant",
    description="Ask questions about agriculture, pest control, fertilizers, and more."
)

# Launch the interface
iface.launch()
