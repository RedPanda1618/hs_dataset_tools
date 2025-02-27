from flask import Flask, render_template, request, jsonify, send_file, Response
import json
import threading
import datetime
from base.hs import *
import io
import webbrowser

app = Flask(__name__)
flow_data = []  # 撮影フローを保持するリスト
status = "完了"
status_index = 0
is_paused = False
wait_event = threading.Event()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/save_flow", methods=["POST"])
def save_flow():
    global flow_data
    try:
        flow_data = json.loads(request.data)["flow"]
        flow_data_json = json.dumps(flow_data, indent=4)
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"flow_{now}.json"
        return Response(
            flow_data_json,
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename={file_name}"},
        )
    except Exception as e:
        print("Error in save_flow:", e)
        return jsonify({"status": "error", "message": str(e)})


@app.route("/run_flow", methods=["POST"])
def run_flow():
    global flow_data

    flow_data = request.json.get("flow")
    thread = threading.Thread(target=execute_flow)
    thread.start()
    return jsonify({"status": "実行中"})


@app.route("/import_flow", methods=["POST"])
def import_flow():
    global flow_data
    try:
        file_content = request.json.get("file_content")
        flow_data = json.loads(file_content)

        return jsonify({"status": "success", "flow": flow_data})

    except Exception as e:
        return jsonify({"status": "error"})


@app.route("/stop_flow", methods=["GET"])
def stop_flow():
    stop_hs()
    return jsonify({"status": "stopped"})


@app.route("/resume_flow", methods=["GET"])
def resume_flow():
    wait_event.set()
    return jsonify({"status": "resumed"})


@app.route("/check_status", methods=["GET"])
def check_status():
    return jsonify({"status": status, "status_index": status_index})


def wait_input():
    global is_paused, status
    is_paused = True
    status = "チェック待ち"
    wait_event.clear()
    wait_event.wait()
    is_paused = False
    status = "実行中"


def execute_flow():
    global flow_data, status, status_index
    status = "実行中"
    print("Flow execution started.")
    print(flow_data)
    print(type(flow_data))
    for i, step in enumerate(flow_data):
        if status == "完了":
            break
        status_index = i
        action = step.get("action")
        params = step.get("params", {})

        if action == "scan":
            scan_hs(params["scan_rate"])
        elif action == "preview":
            preview_hs()
        elif action == "dark":
            dark_scan(params["scan_rate"])
        elif action == "select_window":
            select_window(params["window_num"])
        elif action == "set_scan_rate":
            set_scan_rate(params["scan_rate"])
        elif action == "set_gain":
            set_gain(params["gain"])
        elif action == "sleep":
            sleep(params["duration"])
        elif action == "save":
            save_hs()
        elif action == "delete_dark":
            delete_dark_hs()
        elif action == "wait":
            wait_input()
        elif action == "quit":
            print("Flow execution finished.")
            break
        else:
            print(f"Invalid action: {action}")
            break
    status = "完了"
    status_index = 0
    print("Flow execution finished.")


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
