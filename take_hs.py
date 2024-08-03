from base.hs import *

if __name__ == "__main__":
    high_scan_rate = int(input("Enter high scan rate: "))
    low_scan_rate = int(input("Enter low scan rate: "))
    high_scan_second = int(input("Enter high scan second: "))
    low_scan_second = int(input("Enter low scan second: "))
    save_time = int(input("Enter save time: "))
    cmd_num = int(input("Enter command prompt number: "))
    capture_num = int(input("Enter capture number: "))

    while True:
        print("s: scan, p: preview, d: dark, q: quit")
        key = input("Enter key: ")
        if key == "s":
            select_window(capture_num)
            print("Start scanning")
            set_scan_rate(high_scan_rate)
            scan_hs(high_scan_rate)
            sleep(high_scan_second)
            sleep(save_time)

            set_scan_rate(low_scan_rate)
            scan_hs(low_scan_rate)
            sleep(low_scan_second)
            sleep(save_time)
            select_window(cmd_num)
            print("End scanning")

        elif key == "p":
            select_window(capture_num)
            preview_hs()
            select_window(cmd_num)
        elif key == "d":
            select_window(capture_num)
            black_hs()
            select_window(cmd_num)
        elif key == "q":
            break
