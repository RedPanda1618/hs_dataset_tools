from base.hs import *

if __name__ == "__main__":
    high_scan_rate = int(input("Enter high scan rate: "))
    low_scan_rate = int(input("Enter low scan rate: "))
    high_scan_second = int(input("Enter high scan second: "))
    low_scan_second = int(input("Enter low scan second: "))
    save_time = int(input("Enter save time: "))

    while True:
        key = input("Enter key: ")
        if key == "s":
            scan_hs(high_scan_rate)
            sleep(high_scan_second)

            scan_hs(low_scan_rate)
            sleep(low_scan_second)
        elif key == "p":
            preview_hs()
        elif key == "d":
            black_hs()
        elif key == "q":
            break
