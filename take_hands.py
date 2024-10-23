from time import sleep
from playsound import playsound
from gtts import gTTS
import os
import datetime
from base.hs import *

START_DATETIME = datetime.datetime.now()


def init_voice(hand_positions, hand_positions_text):
    v = []
    voice_dir = "voice_data"
    for i, hand_position_text in enumerate(hand_positions_text):
        text = hand_position_text
        voice = gTTS(text=text, lang="ja")
        file_name = os.path.join(voice_dir, f"{hand_positions[i]}.mp3")
        voice.save(file_name)
        v.append(file_name)
    return v


# ユーザ名の入力関数
def get_username():
    username = input("ユーザ名を入力してください: ")
    print(f"{username}さん、撮影を開始します。")
    return username


# 手の位置の案内をする関数
def guide_hand_position(position, voice):
    # ボイスの再生
    playsound(voice)

    # RGBカメラで確認する処理を想定 (ここでは擬似的に)
    print(f"RGBカメラで{position}の確認完了。")


def wait_for_enter():
    input("準備ができたらEnterキーを押してください。")


# HS撮影を非同期で行う関数
def start_hs_capture(capture_time, done_voice):
    print("HS撮影を開始します...")
    sleep(capture_time)  # 撮影時間を待機
    print("HS撮影が完了しました。")
    playsound(done_voice)


# 保存処理を非同期で行う関数
def save_data(save_time):
    print("データを保存しています...")
    sleep(save_time)  # 保存時間を待機
    print("データの保存が完了しました。")


def save_csv(user_name, position, datetime):
    os.makedirs(f"out/{START_DATETIME}", exist_ok=True)
    with open(f"out/{START_DATETIME}/data.csv", "a") as f:
        f.write(f"{user_name},{position},{datetime}\n")


# メインの撮影ループ
def capture_loop(conf):
    # 撮影する手の位置のリスト（例）
    hand_positions_text = [
        "０番のテープに中指を置いて，掌を下にしてください",
        "位置をそのままに，掌を上にしてください",
        "１番のテープに中指を置いて，掌を下にしてください",
        "位置をそのままに，掌を上にしてください",
        "２番のテープに中指を置いて，掌を下にしてください",
        "位置をそのままに，掌を上にしてください",
        "３番のテープに中指を置いて，掌を下にしてください",
        "位置をそのままに，掌を上にしてください",
        "掌を下にしてください．４番のテープに中指を置いて，指を内側に向けてください",
        "５番のテープに中指を置いて，指を外側に向けてください",
        "左にある黒い台を６番のテープに合わせて設置し，エーの面に手を置いてください．指は正面に向けてください",
        "ビーの面に手を置いてください",
        "シーの面に手を置いてください",
        "ディーの面に手を置いてください",
        "台の先端に手を置いて，水平にしてください．指は正面に向けてください．",
    ]
    hand_positions = list(range(len(hand_positions_text)))
    next_text = "次の撮影に進みます．"
    done_text = "撮影が完了しました．"
    voices = init_voice(hand_positions, hand_positions_text)
    next_voice = init_voice(["next"], [next_text])[0]
    done_voice = init_voice(["done"], [done_text])[0]
    select_window(conf["capture_num"])
    set_scan_rate(conf["scan_rate"])
    while True:
        username = get_username()
        for i, position in enumerate(hand_positions):
            guide_hand_position(hand_positions_text[i], voices[i])
            wait_for_enter()
            save_csv(username, position, datetime.datetime.now())
            select_window(conf["capture_num"])
            start_hs_capture(conf["scan_second"], next_voice),
            save_data(conf["save_time"])
            select_window(conf["cmd_num"]),
        playsound(done_voice)


def main():
    scan_rate = int(input("Enter scan rate: "))
    scan_second = int(input("Enter scan second: "))
    save_time = int(input("Enter save time: "))
    cmd_num = int(input("Enter command prompt number: "))
    capture_num = int(input("Enter capture number: "))

    conf = {
        "scan_rate": scan_rate,
        "scan_second": scan_second,
        "save_time": save_time,
        "cmd_num": cmd_num,
        "capture_num": capture_num,
    }
    capture_loop(conf)


# エントリーポイント
if __name__ == "__main__":
    main()
