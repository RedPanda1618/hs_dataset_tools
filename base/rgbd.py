import os
import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
import datetime

class Realsense:
    def __init__(self, save_path, strat_num):
        # ストリーム(Depth/Color)の設定
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.save_path = save_path
        self.start_num = strat_num

        # 時刻のディレクトリを作成
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(self.save_path, timestamp)
        os.makedirs(self.save_path, exist_ok=True)

        # datetimeの下の階層に'npy'と'png'ディレクトリを作成
        npy_dir = os.path.join(self.save_path, 'npy')
        png_dir = os.path.join(self.save_path, 'png')
        os.makedirs(npy_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        # datetimeの下の階層に'npy'と'png'ディレクトリを作成
        self.npy_save_path = os.path.join(self.save_path, 'npy')
        self.png_save_path = os.path.join(self.save_path, 'png')
        os.makedirs(self.npy_save_path, exist_ok=True)
        os.makedirs(self.png_save_path, exist_ok=True)

    def take_image(self):
        print("Press 's' to save image, 'q' or 'esc' to quit.")
        # ストリーミング開始
        pipeline = rs.pipeline()
        profile = pipeline.start(self.config)
        # Alignオブジェクト生成
        align_to = rs.stream.color
        align = rs.align(align_to)
        counter = self.start_num
        try:
            while True:

                # フレーム待ち(Color & Depth)
                frames = pipeline.wait_for_frames()

                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame or not color_frame:
                    continue

                #imageをnumpy arrayに
                color_image = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())

                #depth imageをカラーマップに変換
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.08), cv2.COLORMAP_JET)

                #画像表示
                color_image_s = cv2.resize(color_image, (640, 480))
                depth_colormap_s = cv2.resize(depth_colormap, (640, 480))
                images = np.hstack((color_image_s, depth_colormap_s))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                key = cv2.waitKey(1) & 0xFF

                #画像保存
                if key == ord('s'):
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    # save rgb
                    cv2.imwrite(f"{self.png_save_path}/{counter:05d}-rgb-{timestamp}.png", color_image_s)
                    color_image_s_rgb = color_image_s[:,:,[2,1,0]]
                    np.save(f"{self.npy_save_path}/{counter:05d}-rgb-{timestamp}.npy", color_image_s_rgb)
                    print(f"Save: {counter:05d}-rgb-{timestamp}")

                    # save depth
                    cv2.imwrite(f"{self.png_save_path}/{counter:05d}-depth-{timestamp}.png", depth_colormap_s)
                    np.save(f"{self.npy_save_path}/{counter:05d}-depth-{timestamp}.npy", depth_data)
                    print(f"Save: {counter:05d}-depth-{timestamp}")
                    counter += 1

                # q or esc が押された場合は終了する
                if key == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

        finally:
            #ストリーミング停止
            pipeline.stop()

def main(args):
    realsense = Realsense(args.save_path, args.start_num)
    realsense.take_image()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_path', type=str, required=True, help='Path of directory to save.')
    parser.add_argument('-n', '--start_num', type=int, required=True, help='Start number of save count.')

    args = parser.parse_args()
    main(args)