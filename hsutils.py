import cv2
import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import numpy as np
import imaging as im
import io
import math
import datetime
import yaml
from hsdformat import COLOR_MATCHING_FUNCTION, CONNFIG


class Preprocessing:
    def __init__(self):
        pass

    def data2rgb(self, hsi):
        hsi = hsi.astype(np.float32) / 4095
        M = np.array(
            [
                [0.41844, -0.15866, -0.08283],
                [-0.09117, 0.25242, 0.01570],
                [0.00092, -0.00255, 0.17858],
            ]
        )

        rgb_white_in_xyz = (0.9642, 1.0000, 0.8249)

        height, width = hsi.shape[:2]
        camera = "nh9"

        lower_limit, upper_limit = (
            CONNFIG[camera]["lower_limit"],
            CONNFIG[camera]["upper_limit"],
        )
        wave_length = np.arange(lower_limit, upper_limit + 1, 5)
        idx_390, idx_835 = int(np.where(wave_length == 390)[0]), int(
            np.where(wave_length == 835)[0]
        )

        hsi_390_830 = hsi[:, :, idx_390:idx_835]  # 390 ~ 830 nmの強度
        # 各ピクセルの(x, y, z)の値を計算
        intensity = hsi_390_830.reshape(-1, hsi_390_830.shape[-1])
        x = np.dot(intensity, COLOR_MATCHING_FUNCTION[:, 1]) / rgb_white_in_xyz[0]
        y = np.dot(intensity, COLOR_MATCHING_FUNCTION[:, 2]) / rgb_white_in_xyz[1]
        z = np.dot(intensity, COLOR_MATCHING_FUNCTION[:, 3]) / rgb_white_in_xyz[2]
        xyz_img = np.vstack((x, y, z)).T.reshape(height, width, 3).astype(np.float32)

        # RGB表色系に変換
        rgb_img = np.dot(xyz_img, M.T).astype(np.float32)
        rgb_img = np.clip(rgb_img, 0, 1) * 255
        rgb_img = rgb_img.astype(np.uint8)
        return rgb_img

    def adjust(self, img, alpha=1.0, beta=0.0, gamma=1.0):
        dst = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        dst = dst.astype(np.uint8)
        img2gamma = np.zeros((256, 1), dtype=np.uint8)

        for i in range(256):
            img2gamma[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
        gamma_img = cv2.LUT(dst, img2gamma)
        return np.clip(gamma_img, 0, 255).astype(np.uint8)

    def load_row_nh9(self, file_path, side=None):
        h = 1080
        w = 2048
        c = 151
        data_origin = np.fromfile(file_path, dtype=np.uint16)
        shape = (w, c, h)
        data = data_origin.reshape(shape, order="F")
        data = data.transpose(2, 0, 1)
        if side == "left":
            data = data[:, : w // 2, :]
        elif side == "right":
            data = data[:, w // 2 :, :]
        elif side is not None:
            print("side is not correct")
        return data

    def white_correction(self, data, white_data):
        corrected_data = data / white_data
        return corrected_data

    def dark_correction(self, data, dark_data):
        data_dtype = data.dtype
        data = data.astype(np.float32)
        dark_data = dark_data.astype(np.float32)
        corrected_data = data - dark_data
        corrected_data = np.where(corrected_data < 0, 0, corrected_data).astype(
            data_dtype
        )
        return corrected_data


class Annotation(Preprocessing):
    def __init__(self):
        self.lbutton = False
        self.out = None
        self.data = None
        self.mask = None
        self.radius = 10
        self.window_name = "SelectAnnotationArea"
        self.wave = 0
        self.magnification = 0.8
        self.mask_value = 200
        self.log = []

    def onclick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.log.append(self.mask.copy())

        x = int(x // self.magnification)
        y = int(y // self.magnification)

        if (event == cv2.EVENT_LBUTTONDOWN) and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            self.lbutton = True
            cv2.circle(
                self.mask,
                center=(x, y),
                radius=self.radius,
                color=(0, 0, 0),
                thickness=-1,
                lineType=cv2.LINE_4,
            )
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.lbutton = True
            cv2.circle(
                self.mask,
                center=(x, y),
                radius=self.radius,
                color=(0, self.mask_value, self.mask_value),
                thickness=-1,
                lineType=cv2.LINE_4,
            )
        elif (
            (event == cv2.EVENT_MOUSEMOVE)
            and (self.lbutton)
            and (flags & cv2.EVENT_FLAG_SHIFTKEY)
        ):
            cv2.circle(
                self.mask,
                center=(x, y),
                radius=self.radius,
                color=(0, 0, 0),
                thickness=-1,
                lineType=cv2.LINE_4,
            )
        elif (event == cv2.EVENT_MOUSEMOVE) and (self.lbutton):
            cv2.circle(
                self.mask,
                center=(x, y),
                radius=self.radius,
                color=(0, self.mask_value, self.mask_value),
                thickness=-1,
                lineType=cv2.LINE_4,
            )
        elif event == cv2.EVENT_LBUTTONUP:
            self.lbutton = False
        if self.mode == "rgb":
            out = cv2.addWeighted(self.data, 0.7, self.mask, 0.3, 0)
        elif self.mode == "hs":
            out = cv2.addWeighted(self.data2img(self.wave), 0.7, self.mask, 0.3, 0)

        if event == event == cv2.EVENT_MOUSEMOVE:
            cv2.circle(
                out,
                center=(x, y),
                radius=self.radius,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_4,
            )

        out = cv2.resize(out, None, fx=self.magnification, fy=self.magnification)
        cv2.imshow(self.window_name, out)

    def on_trackbar_radius(self, val):
        self.radius = val
        center = (self.data.shape[1] // 2, self.data.shape[0] // 2)
        marker = np.zeros(self.data.shape[:2] + (3,), dtype=np.uint8)

        cv2.circle(
            marker,
            center=center,
            radius=self.radius,
            color=(0, 100, 100),
            thickness=-1,
            lineType=cv2.LINE_4,
        )
        if self.mode == "rgb":
            out = cv2.addWeighted(self.data, 0.7, self.mask, 0.3, 0)
        elif self.mode == "hs":
            out = cv2.addWeighted(self.data2img(self.wave), 0.7, self.mask, 0.3, 0)
        out = cv2.addWeighted(out, 0.7, marker, 0.3, 0)
        out = cv2.resize(out, None, fx=self.magnification, fy=self.magnification)
        cv2.imshow(self.window_name, out)

    def on_trackbar_wave(self, val):
        self.wave = val
        out = cv2.addWeighted(self.data2img(self.wave), 0.7, self.mask, 0.3, 0)
        out = cv2.resize(out, None, fx=self.magnification, fy=self.magnification)
        cv2.imshow(self.window_name, out)

    def data2img(self, wave):
        img = self.data[:, :, self.wave]
        # img = img / np.max(img) * 200
        img = img / self.data_max * 200
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return img

    def create_mask(self, f_path, mask_path=None):
        """Support creating mask

        Args:
            f_path (str): file path of nh9 data

        Returns:
            ndarray: mask data(0 or 1)
        """
        if mask_path is None:
            mask_path = f_path[:-4].replace("nh9/", "mask/") + "_mask.npy"
        mask_dir = os.path.dirname(mask_path)

        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        print("mask_path", mask_path)
        self.data = self.load_row_nh9(f_path)
        print("data", self.data.shape)
        self.init_mask(mask_path)
        self.window_name = f_path[f_path.rfind("/") + 1 :]
        if self.data.shape[-1] == 3:
            self.mode = "rgb"
            out = cv2.addWeighted(self.data2img(50), 0.8, self.mask, 0.2, 0)
            out = cv2.resize(out, None, fx=self.magnification, fy=self.magnification)

            cv2.imshow(self.window_name, out)
            cv2.setMouseCallback(self.window_name, self.onclick)
            cv2.createTrackbar(
                "Radius", self.window_name, 0, 200, self.on_trackbar_radius
            )
        else:
            self.mode = "hs"
            self.data_max = np.max(self.data)
            out = cv2.addWeighted(self.data2img(self.wave), 0.8, self.mask, 0.2, 0)
            out = cv2.resize(out, None, fx=self.magnification, fy=self.magnification)

            cv2.imshow(self.window_name, out)
            cv2.setMouseCallback(self.window_name, self.onclick)
            cv2.createTrackbar(
                "Radius", self.window_name, 0, 200, self.on_trackbar_radius
            )
            cv2.createTrackbar(
                "Wave",
                self.window_name,
                0,
                (self.data.shape[-1] - 1),
                self.on_trackbar_wave,
            )

        while True:
            key = cv2.waitKey(1)
            if key == ord("s"):
                cv2.destroyAllWindows()
                break
            elif key == ord("m"):
                out = cv2.resize(
                    self.mask, None, fx=self.magnification, fy=self.magnification
                )
                cv2.imshow(self.window_name, out)
            elif key == ord("h"):
                out = cv2.addWeighted(self.data2img(self.wave), 0.8, self.mask, 0.2, 0)
                out = cv2.resize(
                    out, None, fx=self.magnification, fy=self.magnification
                )
                cv2.putText(
                    out,
                    "Creating Mask Automate...",
                    (100, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    4,
                    (255, 255, 255),
                    5,
                    cv2.LINE_AA,
                )
                cv2.imshow(self.window_name, out)
                cv2.waitKey(1)
                self.auto_mask()

                out = cv2.addWeighted(self.data2img(self.wave), 0.8, self.mask, 0.2, 0)
                out = cv2.resize(
                    out, None, fx=self.magnification, fy=self.magnification
                )
                cv2.imshow(self.window_name, out)
                cv2.waitKey(1)
            elif key == ord("c"):
                self.log.append(self.mask.copy())
                self.mask = np.zeros_like(self.mask)
                out = cv2.addWeighted(self.data2img(self.wave), 0.8, self.mask, 0.2, 0)
                out = cv2.resize(
                    out, None, fx=self.magnification, fy=self.magnification
                )
                cv2.imshow(self.window_name, out)
                cv2.waitKey(1)
            elif key == ord("q"):
                exit()
            elif key == ord("z"):
                try:
                    print(self.mask == self.log[-1])
                    print(len(self.log))
                    self.mask = self.log[-1]
                    self.log = self.log[:-1]
                    print(len(self.log))

                    out = cv2.addWeighted(
                        self.data2img(self.wave), 0.8, self.mask, 0.2, 0
                    )
                    out = cv2.resize(
                        out, None, fx=self.magnification, fy=self.magnification
                    )
                    cv2.imshow(self.window_name, out)
                    cv2.waitKey(1)
                except:
                    pass
        mask = self.mask[:, :, 2]
        mask = np.where(mask > 0, 1, 0)
        # plt.imshow(mask)
        # plt.show()
        np.save(mask_path, mask)
        return mask

    def init_mask(self, mask_path):
        self.mask = np.zeros(self.data.shape[:2] + (3,), dtype=np.uint8)
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            self.mask[:, :, 1] = mask
            self.mask[:, :, 2] = mask
            self.mask = np.where(self.mask > 0, self.mask_value, 0)
            self.mask = self.mask.astype(np.uint8)

    def auto_mask(self):
        self.log.append(self.mask.copy())
        mask = self.mask[:, :, 2]
        ind = np.where(mask > 0)
        masked_spectrum = np.mean(self.data[mask > 0], axis=0)
        self.calc_mask(masked_spectrum, 0.995)

    def calc_mask(self, masked_spectrum, threshold):
        result = np.array([])
        a = np.array([masked_spectrum] * self.data.shape[1])
        for i in range(self.data.shape[0]):
            b = self.data[i]
            result = np.sum(a * b, axis=1) / (
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            )
            new_mask = np.where(
                result > threshold, [self.mask_value], self.mask[i, :, 2]
            )
            self.mask[i, :, 1] = new_mask
            self.mask[i, :, 2] = new_mask
        # kernel = np.ones((3, 3), np.uint8)
        # self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        # self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        # self.mask = cv2.erode(self.mask, kernel, iterations=1)


class SelectArea(Preprocessing):
    def __init__(self):
        self.select_cnt = 0
        self.img = None
        self.xy = []
        self.data = None
        self.white_path = None
        self.dataset_path = None

    def get_white_data_row(self, white_path, mask_path):
        self.data = self.load_row_nh9(os.path.join(self.dataset_path, white_path))
        mask = np.load(os.path.join(self.dataset_path, mask_path))
        white_data_row = self.data[mask > 0]
        return white_data_row

    def select_area(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.select_cnt == 0:
            cv2.drawMarker(self.img, (200, 100), (255, 255, 0), markerSize=30)
            self.xy.append((x * 2, y * 2))
            self.select_cnt += 1
            self.imshow("win1", self.img)
        elif event == cv2.EVENT_LBUTTONDOWN and self.select_cnt == 1:
            cv2.drawMarker(self.img, (200, 100), (255, 255, 0), markerSize=30)
            self.xy.append((x * 2, y * 2))
            cv2.rectangle(
                self.img,
                (int(self.xy[0][0] / 2), int(self.xy[0][1] / 2)),
                (int(self.xy[1][0] / 2), int(self.xy[1][1] / 2)),
                (255, 255, 0),
            )
            self.imshow("win1", self.img)
            self.select_cnt += 1


class DisplayByWavelength:
    def __init__(self):
        self.filter_type = "min-max"
        self.fxy = 0.8

    def on_trackbar(self, val):
        img = self.data[:, :, val]
        img = cv2.resize(img, dsize=None, fx=self.fxy, fy=self.fxy)
        if self.filter_type == "min-max":
            img = img / self.data_max * 256
        elif self.filter_type == "z-score":
            img = (img - np.mean(img)) / np.std(img)
            img = img * 128 + 128
        elif self.filter_type == "clahe":
            img = im.clahe(img)
        elif self.filter_type == "equalize_hist":
            img = im.equalize_hist(img)
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imshow(self.window_title, img)

    def show_hs(self, data, filter_type="min-max"):
        self.data = data
        self.data_max = np.max(self.data)
        self.filter_type = filter_type
        cv2.namedWindow(self.window_title)
        cv2.createTrackbar(
            "Threshold",
            self.window_title,
            0,
            (self.data.shape[-1] - 1),
            self.on_trackbar,
        )

        while True:
            # トラックバーの値を取得
            track_value = cv2.getTrackbarPos("Threshold", self.window_title)
            # 最初の１回目の処理を取得した値で実行
            self.on_trackbar(track_value)
            key = cv2.waitKey(1)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break


class LoadDataset(SelectArea):
    def __init__(self):
        self.white_data = {}

    def load_dataset(self, dataset_path):
        self.dataset_path = dataset_path
        with open(os.path.join(self.dataset_path, "dataset.json"), "r") as f:
            self.dataset_info = json.load(f)

        if "hands" not in self.dataset_info.keys():
            self.dataset_info["hands"] = []
        if "bord" not in self.dataset_info.keys():
            self.dataset_info["bord"] = []

        print("setting white data...")
        white_queryset = self.get_queryset("name=='white'")["bord"]
        for white in white_queryset.iloc():
            white_data_row = self.get_white_data_row(white["nh9"], white["mask"])
            white_data_avg = np.mean(white_data_row, axis=0)
            self.white_data[white["light"]] = white_data_avg
        return self.dataset_info

    def get_queryset(self, query=None):
        """get queryset from query

        Args:
            query (str, optional): query for pandas. Defaults to None.

        Returns:
            pd.DataFrame: querysets
        """
        if query is None and "dark" in self.dataset_info.keys():
            return {
                "hands": pd.DataFrame(self.dataset_info["hands"]),
                "bord": pd.DataFrame(self.dataset_info["bord"]),
                "dark": pd.DataFrame(self.dataset_info["dark"]),
            }
        elif query is None:
            return {
                "hands": pd.DataFrame(self.dataset_info["hands"]),
                "bord": pd.DataFrame(self.dataset_info["bord"]),
            }
        else:
            queryset = {"hands": [], "bord": []}
            df_hands = pd.DataFrame(self.dataset_info["hands"])
            df_bord = pd.DataFrame(self.dataset_info["bord"])

            if not df_hands.empty:
                queryset["hands"] = df_hands.query(query)
            else:
                queryset["hands"] = df_hands
            if not df_bord.empty:
                queryset["bord"] = df_bord.query(query)
            else:
                queryset["bord"] = df_bord
            return queryset

    def get_dataset(
        self,
        label,
        max_indata=None,
        query=None,
        replace=False,
        seed=None,
        white=True,
        task="classification",
        white_data=False,
        white_data_noise=None,
        noise_rate=0.01,
        T=None,
        mode="pixel",
        patch_size=6,
    ):
        """get dataset from query

        Args:
            label (str): label to predict
            max_indata (int, optional): max number to pick for dataset. Defaults to None.
            query (str, optional): query for dataset. Defaults to None.
            replace (bool, optional): replace for picking. Defaults to False.
            seed (int): seed for numpy.
            white (boolean): select True when you want to adaptation whitew correction.
            task (str): select task from ["classification", "match"]

        Returns:
            dict: dataset keys are ["x", "y"]
        """

        np.random.seed(seed)
        dataset = {"x": [], "y": []}
        dataset_trans = {"x": [], "y": []}
        queryset = self.get_queryset(query)["hands"]
        for i in tqdm(range(len(queryset)), "loading data"):
            info = queryset.iloc[i]
            if T is None:
                trans = None
                data_masked = self.get_masked_data(
                    info["nh9"],
                    info["mask"],
                    info["light"],
                    white,
                    trans,
                    mode=mode,
                    patch_size=patch_size,
                )
                data_masked_trans = data_masked
            else:
                trans = None
                data_masked = self.get_masked_data(
                    info["nh9"], info["mask"], info["light"], white, trans
                )
                trans = T[info["light"]]
                data_masked_trans = self.get_masked_data(
                    info["nh9"], info["mask"], info["light"], white, trans
                )
            if max_indata is None:
                dataset.append(data_masked)
                dataset_trans.append(data_masked_trans)
            else:
                ind = np.random.choice(
                    range(len(data_masked)), max_indata, replace=replace
                )
                np.random.seed(None)

                for j in ind:
                    if white_data:
                        white_max = np.max(self.white_data[info["light"]])
                        x = np.zeros((2, data_masked.shape[-1]))
                        x_trans = np.zeros((2, data_masked_trans.shape[-1]))
                        x[0] = data_masked[j]
                        x_trans[0] = data_masked_trans[j]

                        if white_data_noise is not None:
                            x[1] = (
                                self.white_data[info["light"]]
                                + np.random.randn(self.white_data[info["light"]].shape)
                                * white_max
                                * noise_rate
                            )
                            x[1] = (
                                self.white_data[info["light"]]
                                + np.random.randn(self.white_data[info["light"]].shape)
                                * white_max
                                * noise_rate
                            )

                    else:
                        x = data_masked[j]
                        x_trans = data_masked_trans[j]
                    dataset["x"].append(x)
                    dataset["y"].append(info[label])
                    dataset_trans["x"].append(x)
                    dataset_trans["y"].append(info[label])
                np.random.seed(seed)

        dataset["y"] = np.array(dataset["y"])
        dataset_trans["y"] = np.array(dataset_trans["y"])
        labels = np.unique(dataset["y"])
        y = np.full((len(dataset["y"])), -1, dtype=np.float32)
        for i, l in enumerate(labels):
            y[np.where(dataset["y"] == l)] = i
        ind = np.arange(len(dataset["y"]))
        np.random.shuffle(ind)
        dataset["x"] = np.array(dataset["x"])[ind].astype(np.float32)
        dataset["y"] = y[ind]

        dataset_trans["x"] = np.array(dataset_trans["x"])[ind].astype(np.float32)
        dataset_trans["y"] = y[ind]

        if task == "classification":
            np.random.seed(None)
            return dataset, dataset_trans
        elif task == "match":
            np.random.seed(None)
            if white_data:
                dataset["x"] = dataset["x"].reshape(-1, 4, dataset["x"].shape[-1])
            else:
                dataset["x"] = dataset["x"].reshape(-1, 2, dataset["x"].shape[-1])

            y_pair = dataset["y"].reshape(-1, 2)
            dataset["y"] = np.where(y_pair[:, 0] == y_pair[:, 1], 1, 0)
            dataset["y"] = dataset["y"].astype(np.int32)
            dataset = self.balance_dataset(dataset)
            return dataset

    def get_patch(self, data, mask, patch_size):
        stride = patch_size
        patches = []
        for i in range(0, data.shape[0] - patch_size, stride):
            for j in range(0, data.shape[1] - patch_size, stride):
                if np.sum(mask[i : i + patch_size, j : j + patch_size]) == (
                    patch_size**2
                ):
                    patches.append(data[i : i + patch_size, j : j + patch_size])
        return np.array(patches)

    def balance_dataset(self, dataset):
        unique_y, counts_y = np.unique(dataset["y"], return_counts=True)
        print("unique_y", unique_y)
        print("counts_y", counts_y)
        min_count = np.min(counts_y)
        new_dataset = {"x": [], "y": []}
        for y in unique_y:
            indices = np.where(dataset["y"] == y)[0]
            np.random.shuffle(indices)
            indices = indices[:min_count]
            new_dataset["x"].extend(dataset["x"][indices])
            new_dataset["y"].extend(dataset["y"][indices])
        new_dataset["x"] = np.array(new_dataset["x"])
        new_dataset["y"] = np.array(new_dataset["y"])
        return new_dataset

    def get_masked_data(
        self,
        data_path,
        mask_path,
        light,
        white=False,
        T=None,
        mode="pixel",
        patch_size=6,
    ):
        data = self.load_nh9(
            os.path.join(self.dataset_path, data_path), light, white, T
        )
        mask = np.load(os.path.join(self.dataset_path, mask_path))
        if mode == "pixel":
            data_masked = data[mask > 0]
        elif mode == "patch":
            data_masked = self.get_patch(data, mask, patch_size=patch_size)
        return data_masked

    def load_nh9(self, file_path, white_light=None, T=None, dark=False):
        """加工されたnh9データを読み込む

        Args:
            file_path (str): データのパス
            white_light (str, optional): 白板補正する場合. Defaults to None.
            T (ndarray, optional): 変換する場合. Defaults to None.
            dark (bool, optional): 黒レベル補正の有無. Defaults to False.

        Returns:
            _type_: _description_
        """
        data = self.load_row_nh9(os.path.join(self.dataset_path, file_path))
        if dark:
            dark_data_path = self.search_dark_data(file_path)
            dark_data = self.load_row_nh9(
                os.path.join(self.dataset_path, dark_data_path)
            )
            data = self.dark_correction(data, dark_data)

        if T is None and white_light is not None:
            white_data = self.white_data[white_light]
        elif white_light is not None:
            white_data = self.calibrate(self.white_data[white_light], T)

        if not T is None:
            data = self.calibrate(data, T)

        if white_light is not None:
            if not T is None:
                data, data_max = self.min_max_normalize_2d(data, beta=1)
                white_data, white_max = self.min_max_normalize_2d(white_data, beta=1)
            data = self.white_correction(data, white_data)

        return data

    def calibrate(self, data, T):
        data = data[..., : len(T)]
        # data_norm, max_val = self.min_max_normalize_2d(data, beta=0)
        data_calibrated = data * T
        return data_calibrated

    def min_max_normalize_2d(self, data, beta=1e-3):
        min_vals = np.min(data)
        max_vals = np.max(data)
        normalized_data = (data - min_vals) / (max_vals - min_vals) + beta
        return normalized_data, max_vals

    def set_white_data(self, white_data_path, mask_path, light):
        """set white data

        Args:
            white_data_path (str): white data nh9 path
            mask_path (str): white data mask path
        """
        white_row_data = self.get_white_data_row(
            os.path.join(self.dataset_path, white_data_path),
            os.path.join(self.dataset_path, mask_path),
        )
        self.white_data[light] = np.average(white_row_data, axis=0)

    def search_dark_data(self, file_path):
        queryset = self.get_queryset()
        queryset_hands = queryset["hands"]
        queryset_bord = queryset["bord"]
        queryset_dark = queryset["dark"]
        queryset_hands["datetime"] = pd.to_datetime(
            queryset_hands["datetime"], format="%Y/%m/%d_%H:%M:%S"
        )
        queryset_bord["datetime"] = pd.to_datetime(
            queryset_bord["datetime"], format="%Y/%m/%d_%H:%M:%S"
        )
        queryset_dark["datetime"] = pd.to_datetime(
            queryset_dark["datetime"], format="%Y/%m/%d_%H:%M:%S"
        )

        data_time = None
        for i in range(len(queryset_hands)):
            info = queryset_hands.iloc[i]
            if os.path.basename(info["nh9"]) == os.path.basename(file_path):
                data_time = info["datetime"]
        if data_time is None:
            for i in range(len(queryset_bord)):
                info = queryset_bord.iloc[i]
                if os.path.basename(info["nh9"]) == os.path.basename(file_path):
                    data_time = info["datetime"]

        if data_time is None:
            return None
        else:
            dark_path = None
            min_time_diff = datetime.timedelta(days=1)
            for i in range(len(queryset_dark)):
                info = queryset_dark.iloc[i]
                time_diff = abs(data_time - info["datetime"])
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    dark_path = info["nh9"]
            return dark_path


class ConvertLight:
    def __init__(self):
        self.T = {}

    def set_convert_matrix(self, light, T):
        """変換行列をセットする

        Args:
            light (str): 照明の種類
            T (ndarray): 変換行列
        """
        self.T[str(light)] = T

    def convert(self, data, light) -> np.ndarray:
        """データを変換する

        Args:
            data (ndarray): 変換前データ
            light (str): 照明の種類

        Returns:
            ndarray: 変換後データ
        """
        data = data[..., : len(self.T[light])]
        data = data * self.T[light]
        return data


class HSUtils(DisplayByWavelength, LoadDataset):
    def __init__(self):
        super().__init__()
        self.window_title = "HSbyWavelength"
        self.white_data = {}

    def show_nh9(self, data, mode="hs", filter_type="min-max", white=True, T=None):
        """_summary_

        Args:
            data (_type_): npy data. (high, width, channel)
            mode (str, optional): hs or rgb. Defaults to "hs".
            filter_type (str, optional): select from [min-max, clahe, z-score, none]. Defaults to "min-max".
            white (bool, optional): white correction. Defaults to True.
            T (_type_, optional): transformation matrix. Defaults to None.
        """
        if isinstance(data, str):
            data = self.load_nh9(data, white, white, T)

        if mode == "hs":
            self.show_hs(data, filter_type)
        elif mode == "rgb":
            rgb = self.data2rgb(data)
            cv2.imshow("HS RGB image", rgb)
            cv2.waitKey(0)

    def show_average_hist(self, query, white=False, T=None, title=None):
        queryset = self.get_queryset(query)
        handss = queryset["hands"]
        bords = queryset["bord"]
        if len(handss) > 0:
            hists = []
            labels = []
            for i in range(len(handss)):
                info = handss.iloc[i]
                data = self.load_nh9(
                    os.path.join(self.dataset_path, info["nh9"]),
                    info["light"],
                    white,
                    T,
                )
                mask = np.load(os.path.join(self.dataset_path, info["mask"]))
                data_masked = data[mask > 0]
                data_average = np.average(data_masked, axis=0)
                hists.append(data_average)
                labels.append(info["name"] + " " + info["light"] + " " + info["cream"])

            if len(hists) > 0:
                for i in range(len(hists)):
                    plt.plot(
                        np.arange(350, (350 + 5 * len(hists[i])), 5),
                        hists[i],
                        label=labels[i],
                    )
                plt.legend()
                if title is not None:
                    plt.title(title)
                plt.show()

        elif len(bords) > 0:
            hists = []
            labels = []
            for i in range(len(bords)):
                info = bords.iloc[i]
                data = self.load_nh9(
                    os.path.join(self.dataset_path, info["nh9"]),
                    info["light"],
                    white,
                    T,
                )
                mask = np.load(os.path.join(self.dataset_path, info["mask"]))
                data_masked = data[mask > 0]
                data_average = np.average(data_masked, axis=0)
                hists.append(data_average)
                labels.append(info["name"] + " " + info["light"])

            if len(bords) > 0:
                self.show_hists(
                    hists,
                    white,
                    labels,
                    max_range=(350 + 5 * len(hists[i])),
                    title=title,
                )
        else:
            print("No data")
        return np.array(hists), labels

    def show_hists(
        self,
        hists,
        white=True,
        labels=None,
        min_range=350,
        max_range=1101,
        step=5,
        title=None,
        out="show",
        ylim=None,
        grid=True,
    ):
        """複数のデータのヒストグラムをまとめて表示する

        Args:
            hists (ndarray): 1次元のデータ
            white (bool, optional): 泊板補正. Defaults to True.
            labels (list, optional): 複数のデータをプロットするときのそれぞれのラベル. Defaults to None.
            min_range (int, optional): 最小波長. Defaults to 350.
            max_range (int, optional): 最大波長. Defaults to 1101.
            step (int, optional): 波長のステップ. Defaults to 5.
            title (_type_, optional): 表のタイトル. Defaults to None.
            out (str, optional): 出力形式{"show", "img"}. Defaults to "show".
            ylim (list, optional): y軸の範囲. Defaults to None.

        Returns:
            _type_: _description_
        """
        hists = np.array(hists)
        plt.figure(figsize=(6, 4), dpi=100)
        plt.grid(grid)
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Intensity")
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

        if len(hists.shape) < 2:
            hists = np.array([hists])
        if labels is None:
            for i in range(len(hists)):
                plt.plot(np.arange(min_range, max_range, step), hists[i])
            if white:
                plt.ylim([0, 1.5])

            if title is not None:
                plt.title(title)
            if out == "show":
                plt.show()
            elif out == "img":
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                enc = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                dst = cv2.imdecode(enc, 1)
                dst = dst[:, :, ::-1]
                plt.clf()
                plt.close()
                return dst
        else:
            for i in range(len(hists)):
                plt.plot(
                    np.arange(min_range, max_range, step), hists[i], label=labels[i]
                )
            plt.legend()
            if white:
                plt.ylim([0, 1.5])

            if title is not None:
                plt.title(title)
            if out == "show":
                plt.show()
            elif out == "img":
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                enc = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                dst = cv2.imdecode(enc, 1)
                dst = dst[:, :, ::-1]
                plt.clf()
                plt.close()
                return dst
