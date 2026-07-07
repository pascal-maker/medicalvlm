import streamlit as st
import os
import torch
import tempfile
import uuid
from PIL import Image

# Qwen-specific imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Manually override the path to avoid "torch.classes" scanning errors
torch.classes.__path__ = []

# Disable file-watcher warnings in Streamlit
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

@st.cache_resource
def load_model_and_processor(hf_token=None):
    """
    Loads the Qwen2.5-VL-7B-Instruct model and processor.
    Forces usage of MPS if available, else CPU.
    """

    # 1) Pick device: MPS if available, else CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    st.write(f"Using device: {device}")

    # Authorization args if the model is gated (i.e., private or requires license acceptance)
    auth_kwargs = {}
    if hf_token and hf_token.strip():
        auth_kwargs["use_auth_token"] = hf_token

    # 2) Load the model
    #    Qwen2.5-VL-7B-Instruct on MPS with eager attention
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,         # needed for Qwen custom code
        attn_implementation="eager",    # eager attention on MPS
        torch_dtype=torch.float32,      # safer on MPS than bfloat16/float16
        low_cpu_mem_usage=True,         # requires accelerate>=0.26.0
        device_map=None,                # do not use "auto" if you want full control
        **auth_kwargs
    )
    # Manually move model to MPS (if available)
    model.to(device)

    # 3) Load the processor
    #    IMPORTANT: The processor should match the model version (7B if that’s what you are using).
    #    If the 7B model doesn't have a 7B-specific processor published, you might do 3B, 
    #    but typically you match them if they exist. 
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
        **auth_kwargs
    )

    return model, processor, device

class MedicalVLMAgent:
    """
    A Qwen-based Vision+Language model agent specialized for medical Q&A.
    Uses a system prompt disclaiming it's not a licensed medical professional.
    """

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.system_prompt = (
            "You are a medical information assistant with vision capabilities. "
            "Disclaimer: I am not a licensed medical professional. "
            "The information provided is for reference only and should not be taken as medical advice. "
            "If you have serious concerns, consult a healthcare provider."
        )

    def run(self, user_text: str, image: Image.Image = None) -> str:
        """
        Build Qwen-style messages, optionally including user-uploaded image.
        Then run model.generate() and decode output text.
        """
        # Prepare Qwen-style messages
        messages = []

        # (1) Add a system-level disclaimer
        messages.append({
            "role": "system",
            "content": [
                {"type": "text", "text": self.system_prompt}
            ]
        })

        # (2) User message content
        user_content = []
        if image:
            # Save the PIL image to a temporary file
            temp_filename = f"/tmp/{uuid.uuid4()}.png"
            image.save(temp_filename)
            user_content.append({"type": "image", "image": temp_filename})

        # Prevent empty user text => zero token errors
        if not user_text.strip():
            user_text = "Please describe the image or provide some medical context."
        user_content.append({"type": "text", "text": user_text})

        messages.append({
            "role": "user",
            "content": user_content
        })

        # Apply Qwen's chat template
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Gather images/videos for Qwen
        image_inputs, video_inputs = process_vision_info(messages)

        # Build model inputs
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        # Move inputs to MPS (or CPU if device="cpu")
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128
            )

        # Strip input tokens from the generation
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        if output_texts:
            return output_texts[0]
        return "**No output text was generated.**"

def main():
    st.title("Medical Qwen2.5-VL (MPS)")

    # Optional HF token if model is gated
    hf_token = st.text_input("Enter your Hugging Face token (if needed)", type="password")

    # Load the model + processor; default to MPS if available
    model, processor, device = load_model_and_processor(hf_token)
    agent = MedicalVLMAgent(model, processor, device)

    # Streamlit UI
    user_question = st.text_area("Ask a medical question or describe symptoms:")
    uploaded_image = st.file_uploader("Upload a medical image (optional)", type=["jpg", "png", "jpeg"])

    if st.button("Submit"):
        image = Image.open(uploaded_image) if uploaded_image else None
        response = agent.run(user_question, image)
        st.write("**Response:**", response)

if __name__ == "__main__":
    main()
