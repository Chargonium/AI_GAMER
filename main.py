import pyautogui, win32gui, time, win32con, win32process, mss, keyboard

import numpy as np
import cv2
from utils import get_input_blob

PID = 11072

def get_hwnd_from_pid(pid):
    hwnds = []

    def callback(hwnd, _):
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)

        if found_pid == pid and win32gui.IsWindowVisible(hwnd):
            hwnds.append(hwnd)

    win32gui.EnumWindows(callback, None)

    if not hwnds:
        raise Exception("No window found for PID")

    return hwnds[0]

def focus_window(hwnd):
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.05)


if __name__ == "__main__":

    focus_window(get_hwnd_from_pid(PID))

    screenWidth, screenHeight = pyautogui.size()

    print(screenWidth)

    print(screenHeight)


    pyautogui.moveTo(screenWidth/2,screenHeight/2) # Move mouse to center

    #pyautogui.leftClick()                          # Click to ensure we're in the current application. 

    #pyautogui.press('esc')

    time.sleep(3)         # Okay now we're properly in minecraft

    buffers = []

    fps = 24
    frame_time = 1 / fps  # ~0.0417 seconds per frame

    target_height = 240

    print("start")
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor

        for _ in range(24*60):

            buffer = []
            start = time.perf_counter()

            # Capture screen (returns BGRA)
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)  # shape: (H, W, 4), BGRA

            # Convert to grayscale (fast)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            # Resize to target height, preserve aspect ratio
            h, w = gray.shape
            target_width = int(w * (target_height / h))
            resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            # Convert to bytes and append
            buffer = resized.tobytes() + get_input_blob()

            buffers.append(buffer)


            # Sleep to maintain ~24 FPS
            elapsed = time.perf_counter() - start
            print(elapsed)
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("end")

    with open("raw.img", "wb") as file:
        for buffer in buffers:
            file.write(buffer)
            
