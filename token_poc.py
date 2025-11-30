# Demonstration of tokenization using LLaMA 3.2 tokenizer.
# Run this in a Colab cell to take screenshots for your article.

from transformers import AutoTokenizer

# 1. Load the tokenizer
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. Two Marcus sentences
sentences = [
    "800 wasps assemble.",
    "The refrigerator unionized again, Robert."
]

for i, text in enumerate(sentences, start=1):
    print(f"\n===== Sentence {i} =====")
    print("Original text:")
    print(text)

    # 3. Tokenize
    encoded = tokenizer(text)
    print("\nToken IDs:")
    print(encoded["input_ids"])

    # 4. Decode back to text
    decoded = tokenizer.decode(encoded["input_ids"])
    print("\nDecoded back:")
    print(decoded)

