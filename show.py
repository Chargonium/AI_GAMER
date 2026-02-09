import cv2
import numpy as np

# --- CONFIG ---
frame_height = 240     # height used during capture
frame_width = 426      # width used during capture
frame_size = frame_width * frame_height
file_path = "raw.img"

# --- Key mapping for decoding 2-byte blob ---
key_map = {
    0: 'W', 1: 'A', 2: 'S', 3: 'D',
    4: 'Q', 5: 'E', 6: 'SPACE', 7: 'SHIFT',
    8: 'CTRL', 9: 'F', 10: 'F5', 11: 'ESC',
    12: '1', 13: 'LMB', 14: 'MMB', 15: 'RMB'
}

# --- Read raw file ---
with open(file_path, "rb") as f:
    raw_data = f.read()

total_frames = len(raw_data) // (frame_size + 2)  # each frame + 2-byte blob

print(f"Total frames in file: {total_frames}")

# --- Display video with keystrokes ---
for i in range(total_frames):
    start = i * (frame_size + 2)
    end = start + frame_size
    frame_bytes = raw_data[start:end]
    
    # Extract 2-byte input blob
    blob_bytes = raw_data[end:end+2]
    input_value = int.from_bytes(blob_bytes, byteorder='little')
    
    # Decode pressed keys
    pressed_keys = [name for bit, name in key_map.items() if (input_value >> bit) & 1]
    keys_text = ' '.join(pressed_keys) if pressed_keys else 'None'
    
    # Convert frame to NumPy array and reshape
    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((frame_height, frame_width))
    
    # Overlay pressed keys on frame
    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # convert to BGR to draw text
    cv2.putText(display_frame, f"Keys: {keys_text}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow("Raw Video with Keys", display_frame)
    
    # Wait ~24 FPS, quit on 'q'
    if cv2.waitKey(int(1000 / 24)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
