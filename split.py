import json, random

in_path = "dataset_synthetic.jsonl"
train_path = "dataset_train.jsonl"
val_path = "dataset_val.jsonl"

SYSTEM_PROMPT = (
    "You are Marcus, a strange VRChat character. "
    "You speak in short, surreal, glitchy sentences, mixing existential jokes, "
    "VR/game metaphors and weird observations. You never break character."
)

data = []
with open(in_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        data.append(obj)

random.shuffle(data)

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

def convert(records, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        for obj in records:
            user = obj["input"]
            marcus = obj["output"]
            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": marcus},
                ]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

convert(train_data, train_path)
convert(val_data, val_path)

print("Train examples:", len(train_data), "Val examples:", len(val_data))
