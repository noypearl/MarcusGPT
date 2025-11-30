# inspect_data.py

from datasets import load_dataset

train_path = "dataset_train.jsonl"
val_path   = "dataset_val.jsonl"

def main():
    train_ds = load_dataset("json", data_files=train_path, split="train")
    val_ds   = load_dataset("json", data_files=val_path, split="train")

    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))

    print("\nExample train item:")
    print(train_ds[0])

    print("\nMessages of first item:")
    for msg in train_ds[0]["messages"]:
        print(f"{msg['role'].upper()}: {msg['content']}")

if __name__ == "__main__":
    main()
