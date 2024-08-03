from socket import *
import threading


class TCPboth:
    def __init__(self, port: int, ip: str, timeout=60):
        SrcIP = ip  # 受信元IP
        SrcPort = port  # 受信元ポート番号
        self.SrcAddr = (SrcIP, SrcPort)  # アドレスをtupleに格納
        self.BUFSIZE = 4096  # バッファサイズ指定
        self.conn = None
        self.flg = True
        self.timeout = timeout

    def start_server(self):
        # サーバを起動する
        self.th = threading.Thread(target=self._server_th)
        self.th.start()

    def _server_th(self):
        if self.conn is None:
            self.server = self.create_connection(self.SrcAddr[0], self.SrcAddr[1])
            self.server.listen()
            # 接続を待機
            self.conn, self.addr = self.server.accept()
            # タイムアウトを設定
        self.conn.settimeout(self.timeout)
        while self.flg:
            try:
                s = self.conn.recv(self.BUFSIZE)
                self.OnRecv(s)
            except timeout:
                pass
            except:
                # タイムアウト以外の例外なら接続を閉じる
                self.flg = False
                self.conn.close()
                self.conn = None
                print("server closed")

    def create_connection(self, port: int, ip: str):
        # 接続する
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        self.conn = socket(AF_INET, SOCK_STREAM)
        self.conn.bind(self.SrcAddr)
        self.conn.connect((ip, port))

    def OnRecv(self, s):
        # データ受信時の処理
        print(s)

    def send(self, s):
        # データの送信
        if self.conn is not None:
            self.conn.send(s.encode("UTF-8"))
        else:
            print("No connection!!")

    def close(self):
        self.flg = False
        if self.conn is not None:
            self.conn.close()
        self.th.join()


if __name__ == "__main__":
    # サーバを起動
    print("start")
    s1 = TCPboth(9000)
    s1.start_server()
    st = ""
    while st != "end":
        st = input(">>")
        s1.send(st)

    s1.close()
    print("OK")
    del s1
