import pyautogui as pag
from time import sleep


def scan_hs(scan_rate):
    pag.press("s")


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
