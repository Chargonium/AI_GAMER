import pyautogui, win32gui, time, win32con, win32process, mss, keyboard
import win32api
import win32ui
import win32gui_struct

import numpy as np
import cv2
from utils import get_input_blob

import threading
import queue
import zlib
import struct

PID = 11072

fps = 24
frame_time = 1 / fps
target_height = 240
CHUNK_SIZE = 12

buffer_queue = queue.Queue(maxsize=256)

stop_event = threading.Event()
recording_event = threading.Event()

border_hwnd = None


# ------------------------------------------------
# Window Helpers
# ------------------------------------------------
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


# ------------------------------------------------
# Red Border Overlay
# ------------------------------------------------
def create_border_window():
    global border_hwnd

    wc = win32gui.WNDCLASS()
    wc.lpszClassName = "RecordingBorder"
    wc.lpfnWndProc = win32gui.DefWindowProc
    class_atom = win32gui.RegisterClass(wc)

    border_hwnd = win32gui.CreateWindowEx(
        win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST | win32con.WS_EX_TRANSPARENT,
        class_atom,
        None,
        win32con.WS_POPUP,
        0, 0, 0, 0,
        None,
        None,
        None,
        None
    )

    win32gui.SetLayeredWindowAttributes(border_hwnd, 0, 255, win32con.LWA_ALPHA)


def draw_border(hwnd_target):
    if border_hwnd is None:
        return

    rect = win32gui.GetWindowRect(hwnd_target)
    x, y, x2, y2 = rect

    width = x2 - x
    height = y2 - y

    win32gui.SetWindowPos(
        border_hwnd,
        win32con.HWND_TOPMOST,
        x, y, width, height,
        win32con.SWP_SHOWWINDOW
    )

    hdc = win32gui.GetWindowDC(border_hwnd)
    pen = win32gui.CreatePen(win32con.PS_SOLID, 6, win32api.RGB(255, 0, 0))
    old_pen = win32gui.SelectObject(hdc, pen)

    win32gui.Rectangle(hdc, 0, 0, width, height)

    win32gui.SelectObject(hdc, old_pen)
    win32gui.DeleteObject(pen)
    win32gui.ReleaseDC(border_hwnd, hdc)


def hide_border():
    if border_hwnd:
        win32gui.ShowWindow(border_hwnd, win32con.SW_HIDE)


def border_thread(hwnd_target):
    while not stop_event.is_set():

        if recording_event.is_set():
            draw_border(hwnd_target)
        else:
            hide_border()

        time.sleep(0.05)


# ------------------------------------------------
# Recorder Thread
# ------------------------------------------------
def recorder_thread():
    print("Recorder thread ready")

    with mss.mss() as sct:
        monitor = sct.monitors[1]

        while not stop_event.is_set():

            if not recording_event.is_set():
                time.sleep(0.05)
                continue

            start = time.perf_counter()

            sct_img = sct.grab(monitor)
            img = np.array(sct_img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            h, w = gray.shape
            target_width = int(w * (target_height / h))
            resized = cv2.resize(gray, (target_width, target_height))

            buffer = resized.tobytes() + get_input_blob()

            buffer_queue.put(buffer)

            elapsed = time.perf_counter() - start
            print(elapsed)
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# ------------------------------------------------
# Saver Thread
# ------------------------------------------------
def saver_thread():
    print("Saver thread ready")

    chunk_buffers = []

    with open("raw.imgc", "wb") as file:

        while not stop_event.is_set() or not buffer_queue.empty():

            try:
                buffer = buffer_queue.get(timeout=0.1)
                chunk_buffers.append(buffer)
                buffer_queue.task_done()

                if len(chunk_buffers) >= CHUNK_SIZE:
                    write_chunk(file, chunk_buffers)
                    chunk_buffers.clear()

            except queue.Empty:
                continue

        if chunk_buffers:
            write_chunk(file, chunk_buffers)


# ------------------------------------------------
# Compression Helper
# ------------------------------------------------
def write_chunk(file, buffers):
    raw_data = b"".join(buffers)
    compressed = zlib.compress(raw_data, level=1)

    file.write(struct.pack("<I", len(compressed)))
    file.write(compressed)


# ------------------------------------------------
# F5 Toggle Handler
# ------------------------------------------------
def toggle_recording():
    if recording_event.is_set():
        print("Recording stopped")
        recording_event.clear()
    else:
        print("Recording started")
        recording_event.set()


# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":

    hwnd = get_hwnd_from_pid(PID)
    focus_window(hwnd)

    create_border_window()

    keyboard.add_hotkey("F6", toggle_recording)

    screenWidth, screenHeight = pyautogui.size()
    pyautogui.moveTo(screenWidth / 2, screenHeight / 2)

    recorder = threading.Thread(target=recorder_thread, daemon=True)
    saver = threading.Thread(target=saver_thread, daemon=True)
    border = threading.Thread(target=border_thread, args=(hwnd,), daemon=True)

    recorder.start()
    saver.start()
    border.start()

    print("Press F5 to toggle recording. Press ESC to exit.")

    keyboard.wait("shift+esc")

    stop_event.set()

    recorder.join()
    saver.join()

    print("Shutdown complete")
