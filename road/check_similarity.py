import zipfile
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
import os

# 1. ResNetモデルのロード（全結合層を除く）
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()  # 評価モードに設定
model = torch.nn.Sequential(*list(model.children())[:-1])  # 全結合層を除く

# 2. 画像の前処理を定義
preprocess = transforms.Compose(
    [
        transforms.Resize((680, 680)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNetの平均
            std=[0.229, 0.224, 0.225],  # ImageNetの標準偏差
        ),
    ]
)


def extract_features(img_path):
    """画像から特徴量ベクトルを抽出"""
    img = Image.open(img_path).convert("RGB")  # 画像のロード
    img_tensor = preprocess(img).unsqueeze(0)  # バッチ次元の追加
    with torch.no_grad():  # 勾配不要
        features = model(img_tensor).squeeze().numpy()  # 特徴量取得＆次元圧縮
    return features


def extract_features_batch(img_path_list, batch_size=32):
    img_list = []
    for img_path in tqdm(img_path_list, desc="Loading images"):
        img = Image.open(img_path).convert("RGB")
        img_list.append(img)

    img_tensor = torch.stack(
        [preprocess(img) for img in tqdm(img_list, desc="Preprocessing")]
    )
    data_loader = torch.utils.data.DataLoader(
        img_tensor, batch_size=batch_size, shuffle=False
    )

    features = []
    for batch in tqdm(data_loader, desc="Extracting features"):
        with torch.no_grad():
            batch_features = model(batch).squeeze().numpy()
            features += batch_features.tolist()
    features = np.array(features)
    return features


def cosine_similarity_multi(features, rgb_files):
    """すべての組み合わせのコサイン類似度を計算

    Args:
        features (np.ndarray): 特徴量ベクトルの配列
    """
    os.makedirs("out", exist_ok=True)
    inds = list(range(len(features)))
    combinations = [(i, j) for i in inds for j in inds if i < j]
    similarity_dict = {}
    with open("out/tmp.csv", "w") as f:
        f.write("file1,file2,similarity\n")
        cnt = 0
        for i, j in tqdm(combinations, desc="Calculating cosine similarity"):
            similarity = cosine_similarity([features[i]], [features[j]])[0][0]
            similarity_dict[(i, j)] = similarity
            if cnt % 100 == 0:
                for k, v in similarity_dict.items():
                    f.write(f"{rgb_files[k[0]]},{rgb_files[k[1]]},{v}\n")
                similarity_dict = {}
            cnt += 1

    similarity_df = pd.read_csv("out/tmp.csv", header=True)
    similarity_df_sorted = similarity_df.sort_values(2, ascending=False)
    similarity_df_sorted.to_csv("out/similarity.csv", index=False, header=True)
    os.remove("out/tmp.csv")


def search_from_zip(zip_file_path):
    zip_file_path = zip_file_path.replace("file://", "")
    search_files = [".jpg", ".jpeg", ".png"]
    with zipfile.ZipFile(zip_file_path, "r") as z:
        file_list = z.namelist()
        files = [f for f in file_list if any([s in f for s in search_files])]
    return files


def test():
    # 3. 画像から特徴量を抽出
    img_path = "/home/shumpei/anaconda3/pkgs/binaryornot-0.4.4-pyhd3eb1b0_1/info/test/tests/files/lena.jpg"
    img = Image.open(img_path)
    rotated_img = img.rotate(45)
    rotated_img.save("rotated_lena.jpg")
    img1_features = extract_features(img_path)
    img2_features = extract_features("rotated_lena.jpg")

    # 4. コサイン類似度を計算
    similarity = cosine_similarity([img1_features], [img2_features])

    print(f"類似度: {similarity[0][0]:.4f}")

    features = extract_features_batch([img_path, img_path, "rotated_lena.jpg"])
    cosine_similarity_multi(features)
    os.remove("rotated_lena.jpg")


def main():
    import os

    rgb_dir = input("Enter the path to the RGB images: ")

    if rgb_dir.endswith(".zip"):
        rgb_files = search_from_zip(rgb_dir)
    else:
        rgb_files = []
        for root, _, files in os.walk(rgb_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    rgb_files.append(os.path.join(root, f))
    print(f"Number of RGB images: {len(rgb_files)}")

    features = extract_features_batch(rgb_files)
    cosine_similarity_multi(features, rgb_files)


if __name__ == "__main__":
    # test()
    main()
