from base.hs import *
from playsound import playsound

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
            print("Start scanning high scan rate")
            set_scan_rate(high_scan_rate)
            scan_hs(high_scan_rate)
            sleep(high_scan_second)
            sleep(save_time)
            print("Done.")
            playsound("voice_data/done_1st.mp3")
            print("Start scanning low scan rate")
            set_scan_rate(low_scan_rate)
            scan_hs(low_scan_rate)
            sleep(low_scan_second)
            playsound("voice_data/done_scan.mp3")
            sleep(save_time)
            select_window(cmd_num)
            print("Done.")

        elif key == "p":
            select_window(capture_num)
            set_scan_rate(high_scan_rate)
            preview_hs()
            select_window(cmd_num)
        elif key == "d":
            print("Start dark scan with high scan rate")
            select_window(capture_num)
            black_hs(high_scan_rate)
            sleep(high_scan_second / 1080 * 105)
            sleep(15)
            sleep(save_time)
            delete_black_hs()

            print("Start dark scan with low scan rate")
            black_hs(low_scan_rate)
            sleep(low_scan_second / 1080 * 105)
            playsound("voice_data/done_dark.mp3")
            sleep(15)
            sleep(save_time)
            delete_black_hs()
            select_window(cmd_num)
        elif key == "q":
            break
