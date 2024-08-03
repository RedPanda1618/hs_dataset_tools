from .tcp import TCPboth
from .hs import *


class Client(TCPboth):
    def __init__(
        self,
        ip: str,
        port: int,
        hs_save_dir: str,
        high_scan_rate: int,
        low_scan_rate: int,
    ):
        super().__init__(port, ip)
        self.create_connection(port, ip)
        self.start_server()
        self.functions = {
            "scan": scan_hs,
            "preview": preview_hs,
        }

    def OnRecv(self, s):
        # データ受信時の処理
        print(s)
        if s in self.functions:
            self.functions[s]()
        else:
            self.send("No such function!!")
