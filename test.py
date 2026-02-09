from PIL import Image
import numpy as np

def two_channel_detail_preserve(input_path, output_path, target_height=144):
    # Load image and ensure RGB
    img = Image.open(input_path).convert("RGB")
    
    # Resize while preserving aspect ratio
    w, h = img.size
    scale = target_height / h
    target_width = int(w * scale)
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Convert to NumPy array
    arr = np.asarray(img, dtype=np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # Detail-preserving two-channel transformation:
    # Use green as a luminance stabilizer
    new_R = (2 * R + G) / 3
    new_B = (2 * B + G) / 3
    new_G = np.zeros_like(G)  # green channel unused
    
    # Clip to valid range
    out = np.stack([new_R, new_G, new_B], axis=2)
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    # Convert back to image and save
    Image.fromarray(out, 'RGB').save(output_path)

# Example usage
two_channel_detail_preserve("test.png", "output_144p_rbeee.png", target_height=144)
