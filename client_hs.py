from base.client import Client


if __name__ == "__main__":
    while True:
        try:
            host = input("Enter host: ")
            port = int(input("Enter port[8080]: "))
            hs_save_dir = input("Enter save directory: ")
            high_scan_rate = int(input("Enter high scan rate: "))
            low_scan_rate = int(input("Enter low scan rate: "))
            high_scan_second = int(input("Enter high scan second: "))
            low_scan_second = int(input("Enter low scan second: "))

            if port == "":
                port = 8080

            dataset_dir = input("Enter dataset directory: ")
            client = Client(
                host,
                port,
                hs_save_dir,
                high_scan_rate,
                low_scan_rate,
                high_scan_second,
                low_scan_second,
            )
            client.client.send()
        except Exception as e:
            print(e)
            continue
        else:
            break
