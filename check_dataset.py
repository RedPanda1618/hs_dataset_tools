import glob
import cv2
import numpy as np
import os


def check_dataset(rgb_files, nir_files):
    # rgb_nums = [f[-19:-4] for f in rgb_files]
    # nir_nums = [f[-19:-4] for f in nir_files]

    rgb_nums = rgb_files
    nir_nums = nir_files

    rgb_index = np.argsort(rgb_nums)
    nir_index = np.argsort(nir_nums)

    if len(rgb_files) != len(nir_files):
        print("Number of RGB images and NIR images do not match!")
        return

    for i in range(len(rgb_files)):
        print("RGB Image: ", rgb_files[rgb_index[i]])
        print("NIR Image: ", nir_files[nir_index[i]])

        rgb_img = cv2.imread(rgb_files[rgb_index[i]])
        nir_img = cv2.imread(nir_files[nir_index[i]])

        if rgb_img.shape != nir_img.shape:
            print("RGB: ", rgb_img.shape)
            print("NIR: ", nir_img.shape)
            print("Shape of RGB image and NIR image do not match!")
            return
        added_img = cv2.addWeighted(rgb_img, 0.5, nir_img, 0.5, 0)
        concated_img = np.concatenate((rgb_img, nir_img), axis=1)
        concated_img = cv2.resize(concated_img, (0, 0), fx=0.4, fy=0.4)
        cv2.imshow("Concated Image", concated_img)
        cv2.imshow("NIR Image", nir_img)
        cv2.imshow("RGB Image", rgb_img)
        cv2.imshow("Added Image", added_img)
        cv2.waitKey(0)

    return True


def main():
    rgb_dir = input("Enter the path to the RGB images: ")
    nir_dir = input("Enter the path to the NIR images: ")
    # rgb_dir = "/home/shumpei/ダウンロード/rgb_20240909/"
    # nir_dir = "/home/shumpei/ダウンロード/nir_20240909/"
    rgb_files = glob.glob(os.path.join(rgb_dir, "*.jpg"))
    nir_files = glob.glob(os.path.join(nir_dir, "*.jpg"))

    check_dataset(rgb_files, nir_files)


if __name__ == "__main__":
    main()
