from base import hs
import datetime
import os


def main():
    now = datetime.datetime.now()
    file_name = os.path.join("out", now.strftime("%Y%m%d%H%M%S"))
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    print("Capture the hand of the user")
    scan_rate = int(input("Enter scan rate: "))
    scan_second = int(input("Enter scan second: "))
    save_time = int(input("Enter save time: "))
    cmd_num = int(input("Enter command prompt number: "))
    capture_num = int(input("Enter capture number: "))
    while True:
        try:
            print("s: scan, p: preview, d: dark, q: quit")
            key = input("Enter key: ")
            if key == "s":
                hs.select_window(capture_num)
                print("Start scanning")
                hs.set_scan_rate(scan_rate)
                hs.scan_hs(scan_rate)
                hs.sleep(scan_second)
                hs.sleep(save_time)
                print("Done.")
                print("Start scanning")
                hs.set_scan_rate(scan_rate)
                hs.scan_hs(scan_rate)
                hs.sleep(scan_second)
                hs.sleep(save_time)
                print("Done.")
                hs.select_window(cmd_num)
            elif key == "p":
                hs.select_window(capture_num)
                hs.set_scan_rate(scan_rate)
                hs.preview_hs()
                hs.select_window(cmd_num)
            elif key == "d":
                print("Start dark scan")
                hs.select_window(capture_num)
                hs.black_hs(scan_rate)
                hs.sleep(scan_second / 1080 * 105)
                hs.sleep(15)
                hs.sleep(save_time)
                hs.delete_black_hs()
                print("Done.")
                hs.select_window(cmd_num)
            elif key == "q":
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            break
