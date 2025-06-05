#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined Medical-VLM, **SAM-2 automatic masking**, and CheXagent demo.

⭑ Changes ⭑
-----------
1. Fixed SAM-2 installation and import issues
2. Added proper error handling for missing dependencies
3. Made SAM-2 functionality optional with graceful fallback
4. Added installation instructions and requirements check
"""

# ---------------------------------------------------------------------
# Standard libs
# ---------------------------------------------------------------------
import os
import sys
import uuid
import tempfile
import subprocess
import warnings
from threading import Thread

# Environment setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore", message=r".*upsample_bicubic2d.*")

# ---------------------------------------------------------------------
# Third-party libs
# ---------------------------------------------------------------------
import torch
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr

# =============================================================================
# Dependency checker and installer
# =============================================================================
def check_and_install_sam2():
    """Check if SAM-2 is available and attempt installation if needed."""
    try:
        # Try importing SAM-2
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        return True, "SAM-2 already available"
    except ImportError:
        print("SAM-2 not found. Attempting to install...")
        try:
            # Clone SAM-2 repository
            if not os.path.exists("segment-anything-2"):
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/facebookresearch/segment-anything-2.git"
                ], check=True)
            
            # Install SAM-2
            original_dir = os.getcwd()
            os.chdir("segment-anything-2")
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
            os.chdir(original_dir)
            
            # Add to Python path
            sys.path.insert(0, os.path.abspath("segment-anything-2"))
            
            # Try importing again
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            return True, "SAM-2 installed successfully"
            
        except Exception as e:
            print(f"Failed to install SAM-2: {e}")
            return False, f"SAM-2 installation failed: {e}"

# Check SAM-2 availability
SAM2_AVAILABLE, SAM2_STATUS = check_and_install_sam2()
print(f"SAM-2 Status: {SAM2_STATUS}")

# =============================================================================
# SAM-2 imports (conditional)
# =============================================================================
if SAM2_AVAILABLE:
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.modeling.sam2_base import SAM2Base
        from sam2.utils.misc import get_device_index
    except ImportError as e:
        print(f"SAM-2 import error: {e}")
        SAM2_AVAILABLE = False

# =============================================================================
# Qwen-VLM imports & helper
# =============================================================================
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# =============================================================================
# CheXagent imports
# =============================================================================
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# ---------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# =============================================================================
# Qwen-VLM model & agent
# =============================================================================
_qwen_model = None
_qwen_processor = None
_qwen_device = None

def load_qwen_model_and_processor(hf_token=None):
    global _qwen_model, _qwen_processor, _qwen_device
    if _qwen_model is None:
        _qwen_device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[Qwen] loading model on {_qwen_device}")
        auth_kwargs = {"use_auth_token": hf_token} if hf_token else {}
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None,
            **auth_kwargs,
        ).to(_qwen_device)
        _qwen_processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True,
            **auth_kwargs,
        )
    return _qwen_model, _qwen_processor, _qwen_device

class MedicalVLMAgent:
    """Light wrapper around Qwen-VLM with an optional image."""

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.system_prompt = (
            "You are a medical information assistant with vision capabilities.\n"
            "Disclaimer: I am not a licensed medical professional. "
            "The information provided is for reference only and should not be taken as medical advice."
        )

    def run(self, user_text: str, image: Image.Image | None = None) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
        ]
        user_content = []
        if image is not None:
            tmp = f"/tmp/{uuid.uuid4()}.png"
            image.save(tmp)
            user_content.append({"type": "image", "image": tmp})
        user_content.append({"type": "text", "text": user_text or "Please describe the image."})
        messages.append({"role": "user", "content": user_content})

        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt_text],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=128)
        trimmed = out[0][inputs.input_ids.shape[1] :]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()

# =============================================================================
# SAM-2 model + AutomaticMaskGenerator (conditional)
# =============================================================================
def download_sam2_checkpoint():
    """Download SAM-2 checkpoint if not present."""
    checkpoint_dir = "checkpoints"
    checkpoint_file = "sam2.1_hiera_large.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print("Downloading SAM-2 checkpoint...")
        try:
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
            urllib.request.urlretrieve(url, checkpoint_path)
            print("SAM-2 checkpoint downloaded successfully")
        except Exception as e:
            print(f"Failed to download SAM-2 checkpoint: {e}")
            return None
    
    return checkpoint_path

def initialize_sam2():
    """Initialize SAM-2 model and mask generator."""
    if not SAM2_AVAILABLE:
        return None, None
    
    try:
        # Download checkpoint if needed
        checkpoint_path = download_sam2_checkpoint()
        if checkpoint_path is None:
            return None, None
        
        # Config path (you may need to adjust this)
        config_path = "segment-anything-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        if not os.path.exists(config_path):
            config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        device = get_device()
        print(f"[SAM-2] building model on {device}")

        sam2_model = build_sam2(
            config_path,
            checkpoint_path,
            device=device,
            apply_postprocessing=False,
        )

        mask_gen = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=0,
        )
        return sam2_model, mask_gen
    
    except Exception as e:
        print(f"[SAM-2] Failed to initialize: {e}")
        return None, None

# Initialize SAM-2 (conditional)
_sam2_model, _mask_generator = None, None
if SAM2_AVAILABLE:
    _sam2_model, _mask_generator = initialize_sam2()
    if _sam2_model is not None:
        print("[SAM-2] Successfully initialized!")
    else:
        print("[SAM-2] Initialization failed")

def automatic_mask_overlay(image_np: np.ndarray) -> np.ndarray:
    """Generate masks and alpha-blend them on top of the original image."""
    if _mask_generator is None:
        raise RuntimeError("SAM-2 mask generator not initialized")
        
    anns = _mask_generator.generate(image_np)
    if not anns:
        return image_np

    overlay = image_np.copy()
    if overlay.ndim == 2:  # grayscale → RGB
        overlay = np.stack([overlay] * 3, axis=2)

    for ann in sorted(anns, key=lambda x: x["area"], reverse=True):
        m = ann["segmentation"]
        color = np.random.randint(0, 255, 3, dtype=np.uint8)
        overlay[m] = (overlay[m] * 0.5 + color * 0.5).astype(np.uint8)

    return overlay

def tumor_segmentation_interface(image: Image.Image | None):
    """Tumor segmentation interface with proper error handling."""
    if image is None:
        return None, "Please upload an image."
    
    if not SAM2_AVAILABLE:
        return None, "SAM-2 is not available. Please check installation."
    
    if _mask_generator is None:
        return None, "SAM-2 not properly initialized. Check the console for errors."
        
    try:
        img_np = np.array(image.convert("RGB"))
        out_np = automatic_mask_overlay(img_np)
        n_masks = len(_mask_generator.generate(img_np))
        return Image.fromarray(out_np), f"{n_masks} masks found."
    except Exception as e:
        return None, f"SAM-2 error: {e}"

# =============================================================================
# Simple fallback segmentation (when SAM-2 is not available)
# =============================================================================
def simple_segmentation_fallback(image: Image.Image | None):
    """Simple fallback segmentation using basic image processing."""
    if image is None:
        return None, "Please upload an image."
    
    try:
        import cv2
        from skimage import segmentation, color
        
        # Convert to numpy array
        img_np = np.array(image.convert("RGB"))
        
        # Simple watershed segmentation
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Create overlay
        overlay = img_np.copy()
        overlay[sure_fg > 0] = [255, 0, 0]  # Red overlay
        
        # Alpha blend
        result = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)
        
        return Image.fromarray(result), "Simple segmentation applied (SAM-2 not available)"
        
    except Exception as e:
        return None, f"Fallback segmentation error: {e}"

# =============================================================================
# CheXagent set-up
# =============================================================================
try:
    chex_name = "StanfordAIMI/CheXagent-2-3b"
    chex_tok = AutoTokenizer.from_pretrained(chex_name, trust_remote_code=True)
    chex_model = AutoModelForCausalLM.from_pretrained(
        chex_name, device_map="auto", trust_remote_code=True
    )
    chex_model = chex_model.half() if torch.cuda.is_available() else chex_model.float()
    chex_model.eval()
    CHEXAGENT_AVAILABLE = True
except Exception as e:
    print(f"CheXagent not available: {e}")
    CHEXAGENT_AVAILABLE = False
    chex_tok, chex_model = None, None

def get_model_device(model):
    if model is None:
        return torch.device("cpu")
    for p in model.parameters():
        return p.device
    return torch.device("cpu")

def clean_text(text):
    return text.replace("</s>", "")

@torch.no_grad()
def response_report_generation(pil_image_1, pil_image_2):
    """Structured chest-X-ray report (streaming)."""
    if not CHEXAGENT_AVAILABLE:
        yield "CheXagent is not available. Please check installation."
        return
    
    streamer = TextIteratorStreamer(chex_tok, skip_prompt=True, skip_special_tokens=True)
    paths = []
    for im in [pil_image_1, pil_image_2]:
        if im is None:
            continue
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tfile:
            im.save(tfile.name)
            paths.append(tfile.name)

    if not paths:
        yield "Please upload at least one image."
        return

    device = get_model_device(chex_model)
    anatomies = [
        "View",
        "Airway",
        "Breathing",
        "Cardiac",
        "Diaphragm",
        "Everything else (e.g., mediastinal contours, bones, soft tissues, tubes, valves, pacemakers)",
    ]
    prompts = [
        "Determine the view of this CXR",
        *[
            f'Provide a detailed description of "{a}" in the chest X-ray'
            for a in anatomies[1:]
        ],
    ]

    findings = ""
    partial = "## Generating Findings (step-by-step):\n\n"
    for idx, (anat, prompt) in enumerate(zip(anatomies, prompts)):
        query = chex_tok.from_list_format(
            [*[{"image": p} for p in paths], {"text": prompt}]
        )
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        inp = chex_tok.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        generate_kwargs = dict(
            input_ids=inp,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            streamer=streamer,
        )
        Thread(target=chex_model.generate, kwargs=generate_kwargs).start()
        partial += f"**Step {idx}: {anat}...**\n\n"
        for tok in streamer:
            if idx:
                findings += tok
            partial += tok
            yield clean_text(partial)
        partial += "\n\n"
        findings += " "
    findings = findings.strip()

    # Impression
    partial += "## Generating Impression\n\n"
    prompt = f"Write the Impression section for the following Findings: {findings}"
    conv = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": chex_tok.from_list_format([{"text": prompt}])},
    ]
    inp = chex_tok.apply_chat_template(
        conv, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    Thread(
        target=chex_model.generate,
        kwargs=dict(
            input_ids=inp,
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            streamer=streamer,
        ),
    ).start()
    for tok in streamer:
        partial += tok
        yield clean_text(partial)
    yield clean_text(partial)

@torch.no_grad()
def response_phrase_grounding(pil_image, prompt_text):
    """Very simple visual-grounding placeholder."""
    if not CHEXAGENT_AVAILABLE:
        return "CheXagent is not available. Please check installation.", None
    
    if pil_image is None:
        return "Please upload an image.", None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tfile:
        pil_image.save(tfile.name)
        img_path = tfile.name

    device = get_model_device(chex_model)
    query = chex_tok.from_list_format([{"image": img_path}, {"text": prompt_text}])
    conv = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": query},
    ]
    inp = chex_tok.apply_chat_template(
        conv, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    out = chex_model.generate(
        input_ids=inp, do_sample=False, num_beams=1, max_new_tokens=512
    )
    resp = clean_text(chex_tok.decode(out[0][inp.shape[1] :]))

    # simple center box (placeholder)
    w, h = pil_image.size
    cx, cy, sz = w // 2, h // 2, min(w, h) // 4
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle([(cx - sz, cy - sz), (cx + sz, cy + sz)], outline="red", width=3)

    return resp, pil_image

# =============================================================================
# Gradio UI
# =============================================================================
def create_ui():
    """Create the Gradio interface."""
    # Load Qwen model
    try:
        qwen_model, qwen_proc, qwen_dev = load_qwen_model_and_processor()
        med_agent = MedicalVLMAgent(qwen_model, qwen_proc, qwen_dev)
        qwen_available = True
    except Exception as e:
        print(f"Qwen model not available: {e}")
        qwen_available = False
        med_agent = None

    with gr.Blocks(title="Medical AI Assistant") as demo:
        gr.Markdown("# Combined Medical Q&A · SAM-2 Automatic Masking · CheXagent")
        
        # Status information
        with gr.Row():
            gr.Markdown(f"""
            **System Status:**
            - Qwen VLM: {'✅ Available' if qwen_available else '❌ Not Available'}
            - SAM-2: {'✅ Available' if SAM2_AVAILABLE else '❌ Not Available'}
            - CheXagent: {'✅ Available' if CHEXAGENT_AVAILABLE else '❌ Not Available'}
            """)

        # Medical Q&A Tab
        with gr.Tab("Medical Q&A"):
            if qwen_available:
                q_in = gr.Textbox(label="Question / description", lines=3)
                q_img = gr.Image(label="Optional image", type="pil")
                q_btn = gr.Button("Submit")
                q_out = gr.Textbox(label="Answer")
                q_btn.click(fn=med_agent.run, inputs=[q_in, q_img], outputs=q_out)
            else:
                gr.Markdown("❌ Medical Q&A is not available. Qwen model failed to load.")

        # Segmentation Tab
        with gr.Tab("Automatic masking"):
            seg_img = gr.Image(label="Upload medical image", type="pil")
            seg_btn = gr.Button("Run segmentation")
            seg_out = gr.Image(label="Segmentation result", type="pil")
            seg_status = gr.Textbox(label="Status", interactive=False)
            
            if SAM2_AVAILABLE and _mask_generator is not None:
                seg_btn.click(
                    fn=tumor_segmentation_interface,
                    inputs=seg_img,
                    outputs=[seg_out, seg_status],
                )
            else:
                seg_btn.click(
                    fn=simple_segmentation_fallback,
                    inputs=seg_img,
                    outputs=[seg_out, seg_status],
                )

        # CheXagent Tabs
        with gr.Tab("CheXagent – Structured report"):
            if CHEXAGENT_AVAILABLE:
                gr.Markdown("Upload one or two chest X-ray images; the report streams live.")
                cx1 = gr.Image(label="Image 1", image_mode="L", type="pil")
                cx2 = gr.Image(label="Image 2", image_mode="L", type="pil")
                cx_report = gr.Markdown()
                gr.Interface(
                    fn=response_report_generation,
                    inputs=[cx1, cx2],
                    outputs=cx_report,
                    live=True,
                ).render()
            else:
                gr.Markdown("❌ CheXagent structured report is not available.")

        with gr.Tab("CheXagent – Visual grounding"):
            if CHEXAGENT_AVAILABLE:
                vg_img = gr.Image(image_mode="L", type="pil")
                vg_prompt = gr.Textbox(value="Locate the highlighted finding:")
                vg_text = gr.Markdown()
                vg_out_img = gr.Image()
                gr.Interface(
                    fn=response_phrase_grounding,
                    inputs=[vg_img, vg_prompt],
                    outputs=[vg_text, vg_out_img],
                ).render()
            else:
                gr.Markdown("❌ CheXagent visual grounding is not available.")

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)