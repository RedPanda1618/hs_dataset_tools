import pandas as pd
import json


def json2csv(json_file, csv_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)


def main():
    input_file = input("Enter the path to the JSON file: ").strip()
    if not input_file.endswith(".json"):
        print("Please enter a valid JSON file.")
    output_file = input_file[:-5] + ".csv"
    json2csv(input_file, output_file)


if __name__ == "__main__":
    main()
