import jsonlines
import csv

def load_gold_and_weak_data(jsonl_path, tsv_path):
    data = []

    try:
        with jsonlines.open(jsonl_path) as reader:
            for obj in reader:
                if obj.get("label") is not None:
                    data.append({
                        "transcript": obj["transcript"],
                        "label": obj["label"]
                    })
    except FileNotFoundError:
        print(f"Missing {jsonl_path}, skipping gold-labeled data.")

    try:
        with open(tsv_path, encoding='utf-8') as f:
            tsv_reader = csv.DictReader(f, delimiter="\t")
            for row in tsv_reader:
                if "transcript" in row and "tone_label" in row:
                    data.append({
                        "transcript": row["transcript"],
                        "label": int(row["tone_label"])
                    })
    except FileNotFoundError:
        print(f"Missing {tsv_path}, skipping weak-labeled data.")

    return data