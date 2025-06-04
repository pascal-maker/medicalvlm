#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined Medical-VLM, **SAM-2 automatic masking**, and CheXagent demo.

⭑ Changes ⭑
-----------
1. All Segment-Anything-v1 fallback code has been removed.  
2. A single **SAM-2 AutomaticMaskGenerator** is built once and reused.  
3. Tumor-segmentation tab now runs *fully automatic* masking — no bounding-box textbox.  
4. Fixed SAM-2 config path to use relative path instead of absolute path.
"""

# ---------------------------------------------------------------------
# Standard libs
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
import os, warnings
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"          # CPU fallback for missing MPS ops
warnings.filterwarnings("ignore", message=r".*upsample_bicubic2d.*")  # hide one-line notice

import os
import sys
import uuid
import tempfile
from threading import Thread

# ---------------------------------------------------------------------
# Third-party libs


# ---------------------------------------------------------------------
import torch
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr

# If you cloned facebookresearch/sam2 into the repo root, make sure it's importable
sys.path.append(os.path.abspath("."))

# =============================================================================
# Qwen-VLM imports & helper
# =============================================================================
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# =============================================================================
# SAM-2 imports  (only SAM-2, no v1 fallback)
# =============================================================================
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Alternative: try direct model loading if build_sam2 continues to fail
try:
    from sam2.modeling.sam2_base import SAM2Base
    from sam2.utils.misc import get_device_index
except ImportError:
    print("Could not import additional SAM2 components")


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
# SAM-2 model + AutomaticMaskGenerator
# =============================================================================

# =============================================================================
# SAM-2.1 model + AutomaticMaskGenerator (concise version)
# =============================================================================
# =============================================================================
# SAM-2.1 model + AutomaticMaskGenerator  (final minimal version)
# =============================================================================
import os
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def initialize_sam2():
    # These two files are already in your repo
    CKPT = "checkpoints/sam2.1_hiera_large.pt"   # ≈2.7 GB
    CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # One chdir so Hydra's search path starts inside sam2/sam2/
    os.chdir("sam2/sam2")

    device = get_device()
    print(f"[SAM-2] building model on {device}")

    sam2_model = build_sam2(
        CFG,        # relative to sam2/sam2/
        CKPT,       # relative after chdir
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


# ---------------------- build once ----------------------
try:
    _sam2_model, _mask_generator = initialize_sam2()
    print("[SAM-2] Successfully initialized!")
except Exception as e:
    print(f"[SAM-2] Failed to initialize: {e}")
    _sam2_model, _mask_generator = None, None

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
    if image is None:
        return None, "Please upload an image."
    
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
# CheXagent set-up  (unchanged)
# =============================================================================
chex_name = "StanfordAIMI/CheXagent-2-3b"
chex_tok = AutoTokenizer.from_pretrained(chex_name, trust_remote_code=True)
chex_model = AutoModelForCausalLM.from_pretrained(
    chex_name, device_map="auto", trust_remote_code=True
)
chex_model = chex_model.half() if torch.cuda.is_available() else chex_model.float()
chex_model.eval()


def get_model_device(model):
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def clean_text(text):
    return text.replace("</s>", "")


@torch.no_grad()
def response_report_generation(pil_image_1, pil_image_2):
    """Structured chest-X-ray report (streaming)."""
    streamer = TextIteratorStreamer(chex_tok, skip_prompt=True, skip_special_tokens=True)
    paths = []
    for im in [pil_image_1, pil_image_2]:
        if im is None:
            continue
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tfile:
            im.save(tfile.name)
            paths.append(tfile.name)

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
qwen_model, qwen_proc, qwen_dev = load_qwen_model_and_processor()
med_agent = MedicalVLMAgent(qwen_model, qwen_proc, qwen_dev)

with gr.Blocks() as demo:
    gr.Markdown("# Combined Medical Q&A · SAM-2 Automatic Masking · CheXagent")

    # ---------------------------------------------------------
    with gr.Tab("Medical Q&A"):
        q_in = gr.Textbox(label="Question / description", lines=3)
        q_img = gr.Image(label="Optional image", type="pil")
        q_btn = gr.Button("Submit")
        q_out = gr.Textbox(label="Answer")
        q_btn.click(fn=med_agent.run, inputs=[q_in, q_img], outputs=q_out)

    # ---------------------------------------------------------
    with gr.Tab("Automatic masking (SAM-2)"):
        seg_img = gr.Image(label="Image", type="pil")
        seg_btn = gr.Button("Run segmentation")
        seg_out = gr.Image(label="Overlay", type="pil")
        seg_status = gr.Textbox(label="Status", interactive=False)
        seg_btn.click(
            fn=tumor_segmentation_interface,
            inputs=seg_img,
            outputs=[seg_out, seg_status],
        )

    # ---------------------------------------------------------
    with gr.Tab("CheXagent – Structured report"):
        gr.Markdown("Upload one or two images; the report streams live.")
        cx1 = gr.Image(label="Image 1", image_mode="L", type="pil")
        cx2 = gr.Image(label="Image 2", image_mode="L", type="pil")
        cx_report = gr.Markdown()
        gr.Interface(
            fn=response_report_generation,
            inputs=[cx1, cx2],
            outputs=cx_report,
            live=True,
        ).render()

    # ---------------------------------------------------------
    with gr.Tab("CheXagent – Visual grounding"):
        vg_img = gr.Image(image_mode="L", type="pil")
        vg_prompt = gr.Textbox(value="Locate the highlighted finding:")
        vg_text = gr.Markdown()
        vg_out_img = gr.Image()
        gr.Interface(
            fn=response_phrase_grounding,
            inputs=[vg_img, vg_prompt],
            outputs=[vg_text, vg_out_img],
        ).render()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)