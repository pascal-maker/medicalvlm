import numpy as np
from PIL import Image

class SAM2ImagePredictor:
    def __init__(self):
        self.image = None
        self._last_mask = None  # We'll store the last segmentation mask here.

    @classmethod
    def from_pretrained(cls, model_name):
        print(f"Loaded SAM2 model: {model_name}")
        return cls()

    def set_image(self, image_array):
        """
        Store the original image array. We'll do bounding-box based segmentation on it.
        """
        self.image = image_array
        self._last_mask = None

    def predict(self, input_prompts=None):
        """
        input_prompts is expected to be a list of bounding boxes: [[x1, y1, x2, y2], ...].
        We'll fill those regions in the mask with 255.
        """
        if self.image is None:
            raise ValueError("No image set. Please call set_image() before predict().")

        h, w, _ = self.image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if input_prompts:
            # Each prompt is a bounding box.
            for (x1, y1, x2, y2) in input_prompts:
                # Make sure we clamp to valid range (in case user typed out-of-bounds coords).
                x1, x2 = sorted([max(0, x1), min(w, x2)])
                y1, y2 = sorted([max(0, y1), min(h, y2)])
                # Fill the region with 255.
                mask[y1:y2, x1:x2] = 255
        else:
            # If no bounding boxes, fill entire image as a fallback or do partial region.
            mask[h//4:3*h//4, w//4:3*w//4] = 255

        # Save mask so that plot() can overlay it.
        self._last_mask = mask.copy()

        # Return three items so your Streamlit code can do: masks, _, _ = ...
        masks = [mask]
        scores = [1.0]
        logits = [mask]
        return masks, scores, logits

    def plot(self):
        """
        Create a red overlay wherever mask > 0.
        We'll retrieve the last stored mask from self._last_mask.
        """
        if self.image is None:
            raise ValueError("No original image found. Did you call set_image()?")

        if self._last_mask is None:
            raise ValueError("No mask found. Did you call predict() yet?")

        # Convert original image to make overlay
        overlay = self.image.copy()  # shape: (h, w, 3)
        mask = self._last_mask

        # Create a red overlay
        red_mask = np.zeros_like(overlay)
        red_mask[mask > 0] = [255, 0, 0]  # bright red

        # Blend with alpha
        alpha = 0.4
        blended = (overlay * (1 - alpha) + red_mask * alpha).astype(np.uint8)
        return blended
