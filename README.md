---
title: Medical Vlm Sam2
emoji: 📉
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.32.1
app_file: app.py
pinned: false
short_description: All-in-one medical imaging demo
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Medical VLM with SAM-2 and CheXagent

A comprehensive medical imaging analysis tool that combines:
- Qwen-VLM for medical visual question answering
- SAM-2 (Segment Anything Model 2) for automatic medical image segmentation
- CheXagent for structured chest X-ray report generation

## Features

1. **Medical Q&A**: Ask questions about medical images using the Qwen-VLM model
2. **Automatic Masking**: Segment medical images automatically using SAM-2
3. **Structured Report Generation**: Generate detailed chest X-ray reports using CheXagent
4. **Visual Grounding**: Locate specific findings in medical images

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-vlm-sam2.git
cd medical-vlm-sam2
```

2. Create and activate a virtual environment:
```bash
python -m venv chexagent_env
source chexagent_env/bin/activate  # On Windows: chexagent_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required model checkpoints:
- SAM-2 checkpoint: Place in `checkpoints/sam2.1_hiera_large.pt`
- Other model weights will be downloaded automatically on first run

## Usage

Run the Gradio interface:
```bash
python app.py
```

The web interface will be available at `http://localhost:7860`

## Requirements

- Python 3.8+
- PyTorch
- CUDA-compatible GPU (recommended)
- See `requirements.txt` for full list of dependencies

## License

[Your chosen license]

## Acknowledgments

- [Qwen-VLM](https://github.com/QwenLM/Qwen-VL)
- [SAM-2](https://github.com/facebookresearch/segment-anything)
- [CheXagent](https://github.com/stanfordmlgroup/CheXagent)
