import pyautogui as pag
from time import sleep


def scan_hs(scan_rate=None):
    if scan_rate is not None:
        set_scan_rate(scan_rate)
    pag.press("s")


def select_window(window_num):
    pag.hotkey("win", "d")
    pag.hotkey("win", str(window_num))
    sleep(0.2)


def preview_hs():
    pag.press("p")


def set_scan_rate(scan_rate):
    pag.hotkey("ctrl", "p")
    sleep(0.1)
    pag.typewrite(str(scan_rate))
    sleep(0.1)
    pag.press("enter")
    sleep(0.1)
    pass


def set_gain(gain):
    pag.hotkey("ctrl", "p")
    sleep(0.1)
    pag.press("tab")
    pag.press("tab")
    pag.typewrite(str(gain))
    sleep(0.1)
    pag.press("enter")
    sleep(0.1)
    pass


def save_hs():
    pass


def dark_scan(scan_rate=None):
    if scan_rate is not None:
        set_scan_rate(scan_rate)
    pag.hotkey("ctrl", "d")


def delete_dark_hs():
    pag.hotkey("ctrl", "r")


def stop_hs():
    pag.hotkey("ctrl", "q")
