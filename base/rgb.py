import cv2
import datetime
import platform
import os


class RGBcamera:
    def __init__(self, camera_id, width, height, fps):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self):
        if platform.system() == "Linux":
            self.cap = cv2.VideoCapture(self.camera_id)
        elif platform.system() == "Windows":
            try:
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            except TypeError:
                self.cap = cv2.VideoCapture(self.camera_id)
        else:
            print("Use Linux or Windows.")
            return None

        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)

    def save_img(self, save_dir, cnt, file_type):
        frame = self.cap.read()[1]
        now = datetime.datetime.now()
        now = str(now).replace(" ", "").replace(":", "_").replace(".", "_")
        save_dir = os.path.join(save_dir, now)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, str(cnt) + "." + file_type), frame)

    def show(self):
        frame = self.cap.read()[1]
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            return

    def stop(self):
        self.cap.release()
