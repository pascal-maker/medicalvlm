import gradio as gr
from transformers import pipeline
from PIL import Image
import base64
import io

# Use a pipeline as a high-level helper
pipe = pipeline("image-text-to-text", model="google/medgemma-4b-it")

def ask_about_image(image, question):
    if image is None or question.strip() == "":
        return "Please provide both an image and a question."
    
    try:
        # Convert PIL image to base64 string
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        base64_image = f"data:image/jpeg;base64,{img_str}"
        
        # Create messages in the exact format from your working example
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": base64_image},
                    {"type": "text", "text": question}
                ]
            },
        ]
        
        # Use the pipeline with text parameter as in your example
        result = pipe(text=messages)
        
        # Extract the response
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No response generated")
        elif isinstance(result, dict):
            return result.get("generated_text", "No response generated")
        else:
            return str(result)
            
    except Exception as e:
        return f"Error processing request: {str(e)}"

# Create a single interface with the working method
demo = gr.Interface(
    fn=ask_about_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(lines=2, placeholder="Ask a medical question about the image...", label="Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Med-Gemma Image Analysis",
    description="Upload an image and ask a medical question about it using Google's Med-Gemma 4B model.",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()