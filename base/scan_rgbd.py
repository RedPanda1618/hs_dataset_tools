import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import datetime


class DepthCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)

    def show(self):
        pipeline = rs.pipeline()
        pipeline.start(self.config)
        vis = o3d.visualization.Visualizer()
        vis.create_window("PCD", width=1280, height=720)
        pointcloud = o3d.geometry.PointCloud()
        geom_added = False
        while True:
            dt0 = datetime.datetime.now()
            frames = pipeline.wait_for_frames()
            frames = self.align.process(frames)
            profile = frames.get_profile()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            pcd = self.frame2points(depth_image, color_image_rgb, profile)

            pointcloud.points = pcd.points
            pointcloud.colors = pcd.colors

            if geom_added == False:
                vis.add_geometry(pointcloud)
                geom_added = True
            cv2.imshow("BGR", color_image)

            vis.update_geometry(pointcloud)
            vis.poll_events()
            vis.update_renderer()

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("s"):
                np.savetxt("./depth_data.csv", np.asanyarray(depth_frame.get_data()))

            process_time = datetime.datetime.now() - dt0
            print("\rFPS: " + str(1 / process_time.total_seconds()), end="")

        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()

    def frame2points(self, depth_image, color_image_rgb, profile):

        img_depth = o3d.geometry.Image(depth_image)
        img_color_rgb = o3d.geometry.Image(color_image_rgb)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            img_color_rgb, img_depth, convert_rgb_to_intensity=False
        )

        intrinsics = profile.as_video_stream_profile().get_intrinsics()

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.ppx,
            intrinsics.ppy,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, pinhole_camera_intrinsic
        )

        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return pcd

if __name__ == "__main__":
    camera = DepthCamera()
    camera.show()