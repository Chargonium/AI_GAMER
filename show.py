import cv2
import numpy as np
import zlib
import struct

# --- CONFIG ---
frame_height = 240
frame_width = 426
fps = 24
file_path = "raw.imgc"

frame_size = frame_width * frame_height
buffer_size = frame_size + 2  # frame + 2-byte input blob


# --- Key mapping ---
key_map = {
    0: 'W', 1: 'A', 2: 'S', 3: 'D',
    4: 'Q', 5: 'E', 6: 'SPACE', 7: 'SHIFT',
    8: 'CTRL', 9: 'F', 10: 'F5', 11: 'ESC',
    12: '1', 13: 'LMB', 14: 'MMB', 15: 'RMB'
}


# ---------------------------------------------------
# Pass 1 — Count frames
# ---------------------------------------------------
def count_frames():
    total = 0

    with open(file_path, "rb") as f:
        while True:
            size_bytes = f.read(4)
            if not size_bytes:
                break

            size = struct.unpack("<I", size_bytes)[0]
            compressed = f.read(size)

            raw = zlib.decompress(compressed)
            total += len(raw) // buffer_size

    return total


# ---------------------------------------------------
# Pass 2 — Playback
# ---------------------------------------------------
def playback():
    with open(file_path, "rb") as f:

        while True:
            size_bytes = f.read(4)
            if not size_bytes:
                break

            size = struct.unpack("<I", size_bytes)[0]
            compressed = f.read(size)

            raw = zlib.decompress(compressed)

            # Iterate buffers inside chunk
            chunk_frames = len(raw) // buffer_size

            for i in range(chunk_frames):

                start = i * buffer_size
                end = start + frame_size

                frame_bytes = raw[start:end]
                blob_bytes = raw[end:end+2]

                input_value = int.from_bytes(blob_bytes, byteorder='little')

                pressed_keys = [
                    name for bit, name in key_map.items()
                    if (input_value >> bit) & 1
                ]

                keys_text = ' '.join(pressed_keys) if pressed_keys else 'None'

                frame = np.frombuffer(frame_bytes, dtype=np.uint8)\
                    .reshape((frame_height, frame_width))

                display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                cv2.putText(
                    display_frame,
                    f"Keys: {keys_text}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Playback", display_frame)

                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    return


# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":

    print("Scanning file...")

    total_frames = count_frames()
    total_seconds = total_frames / fps

    print(f"Frames: {total_frames}")
    print(f"Length: {total_seconds:.2f} seconds")

    print("Starting playback... (press Q to quit)")

    playback()

    cv2.destroyAllWindows()
