from hsutils import *
import numpy as np
import glob


def view_npyhsi(file_path):
    hsi = np.load(file_path)
    hsi = hsi.astype(np.float32)
    hsu = HSUtils()
    hsu.show_nh9(hsi, filter_type="min-max")


def main():
    folder_path = input("Enter the path to the folder containing the npyhsi file: ")
    files = glob.glob(folder_path + "/*.npy")

    for i, file in enumerate(files):
        print(f"{i}: {file}")
        view_npyhsi(file)


if __name__ == "__main__":
    main()
