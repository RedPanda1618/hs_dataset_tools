from .tcp import TCPboth
from .hs import *
from time import sleep


class Client(TCPboth):
    def __init__(
        self,
        ip: str,
        port: int,
        hs_save_dir: str,
        high_scan_rate: int,
        low_scan_rate: int,
        high_scan_second: int,
        low_scan_second: int,
    ):
        super().__init__(port, ip)
        self.create_connection(port, ip)
        self.start_server()
        self.functions = {
            "s": scan_hs,
            "b": preview_hs,
            "d": black_hs,
        }
        self.__high_scan_rate = high_scan_rate
        self.__low_scan_rate = low_scan_rate
        self.__high_scan_second = high_scan_second
        self.__low_scan_second = low_scan_second
        self.__hs_save_dir = hs_save_dir

    def OnRecv(self, s):
        # データ受信時の処理
        print(s)
        if s in self.functions:
            if s == "s":
                self.functions[s](self.__high_scan_rate)
                sleep(self.__high_scan_second + 3)
                self.functions[s](self.__low_scan_rate)
                sleep(self.__low_scan_second + 3)

            elif s == "p":
                self.functions[s]()
                pass

        else:
            self.send("No such function!!")
