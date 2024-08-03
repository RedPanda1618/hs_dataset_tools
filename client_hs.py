from base import InetClient


class ClientHS:
    def __init__(self, host: str, port: int):
        self.client = InetClient(host, port)


if __name__ == "__main__":
    while True:
        try:
            host = input("Enter host: ")
            port = int(input("Enter port[8080]: "))
            if port == "":
                port = 8080

            dataset_dir = input("Enter dataset directory: ")
            client = ClientHS(host, port)
            client.client.send()
        except Exception as e:
            print(e)
            continue
        else:
            break
