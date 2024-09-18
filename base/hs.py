import pyautogui as pag
from time import sleep


def scan_hs(scan_rate):
    set_scan_rate(scan_rate)
    pag.press("s")


def select_window(window_num):
    pag.hotkey("win", str(window_num))
    sleep(0.5)


def preview_hs():
    pag.press("p")


def set_scan_rate(scan_rate):
    pag.hotkey("ctrl", "p")
    sleep(0.5)
    pag.typewrite(str(scan_rate))
    sleep(0.5)
    pag.press("enter")
    sleep(0.5)
    pass


def save_hs():
    pass


def black_hs(scan_rate):
    set_scan_rate(scan_rate)
    pag.hotkey("ctrl", "d")
