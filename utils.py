import cv2
import numpy as np
from PIL import Image
import keyboard
import mouse

def get_input_blob():
    """
    Returns a 2-byte blob representing the current state of selected keys/buttons.
    
    Key mapping (bit positions):
    0: w, 1: a, 2: s, 3: d
    4: q, 5: e, 6: space, 7: shift
    8: ctrl, 9: f, 10: f5, 11: esc
    12: 1, 13: left mouse, 14: middle mouse, 15: right mouse
    """
    value = 0
    
    # Keyboard mapping
    key_map = {
        'w': 0, 'a': 1, 's': 2, 'd': 3,
        'q': 4, 'e': 5, 'space': 6, 'shift': 7,
        'ctrl': 8, 'f': 9, 'f5': 10, 'esc': 11,
        '1': 12
    }

    # Mouse mapping
    button_map = {
        'left': 13,
        'middle': 14,
        'right': 15
    }

    # Poll keyboard
    for k, bit in key_map.items():
        if keyboard.is_pressed(k):
            value |= (1 << bit)

    # Poll mouse buttons
    if mouse.is_pressed(button='left'):
        value |= (1 << button_map['left'])
    if mouse.is_pressed(button='middle'):
        value |= (1 << button_map['middle'])
    if mouse.is_pressed(button='right'):
        value |= (1 << button_map['right'])

    # Return 2-byte little-endian blob
    return value.to_bytes(2, byteorder='little')

def convert_image(img: Image.Image, target_height=144) -> Image.Image:
    # Resize
    w, h = img.size
    target_width = int(w * (target_height / h))
    img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)

    return img.convert("L")
