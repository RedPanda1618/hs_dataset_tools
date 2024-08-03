from .tcp import TCPboth
from .rgbd import Realsense
from .rgb import RGBcamera


class Host(TCPboth):
    def __init__(
        self,
        ip: str,
        port: int,
        cameraid: int,
        width: int,
        height: int,
        fps: int,
    ):
        super().__init__(port, ip)
        self.start_server()
        self.rgb = RGBcamera(camera_id=cameraid, width=width, height=height, fps=fps)
        self.rgb.start()

    def wait(self):
        st = ""
        cnt = 0
        while st != "end":
            st = input(">>")
            if st == "s":
                self.rgb.save_img("img", cnt, "png")
                cnt += 1
            self.tcp.send(st)
        self.tcp.close()

    def OnRecv(self, s):
        # データ受信時の処理
        print(s)
