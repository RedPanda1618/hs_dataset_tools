from base.host import Host


if __name__ == "__main__":
    while True:
        try:
            host_ip = input("Enter host IP: ")
            host_port = int(input("Enter host port [8080]: "))
            save_dir = input("Enter save directory for RGBD: ")
            camera_id = int(input("Enter camera ID [0]: "))
            width = int(input("Enter width [640]: "))
            height = int(input("Enter height [480]: "))
            fps = int(input("Enter FPS [30]: "))
            if save_dir == "":
                save_dir = "img"
            if camera_id == "":
                camera_id = 0
            if width == "":
                width = 640
            if height == "":
                height = 480
            if fps == "":
                fps = 30
            if host_port == "":
                host_port = 8080

            host = Host(
                ip=host_ip,
                port=host_port,
                cameraid=0,
                width=640,
                height=480,
                fps=30,
            )
            host.wait()

        except Exception as e:
            print(e)

        except KeyboardInterrupt:
            break
        else:
            host.close()
