import os
import time
import shutil
import tiktoken
from groq import Groq

# === API Keys ===
API_KEYS = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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"
]

# === Model List ===
MODEL_LIST = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "deepseek-r1-distill-llama-70b",
    "compound-beta-mini",
    "compound-beta",
    "llama-guard-3-8b",
    "whisper-large-v3-turbo",
    "qwen-qwq-32b",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "llama3-8b-8192"
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "mistral-saba-24b",
    "meta-llama/llama-4-maverick-17b-128e-instruct"
]

# === Prompt ===
REWRITE_PROMPT = """The stereotype is written in S and their reality is written in R. If you find any slight similar stereotype in the text then write like:
S: (The exact stereotypic sentence/s from the text and strictly keep the sentence exact same).
R: (The reality sentence/s that should be augmented after the stereotypic sentence).
No extra comments or nothing else. Just the stereotype and reality in S: and R: format. Leran from the following examples and follow the instruction properly.


S: Women are more nurturing and better suited to care for others.
R: People of all genders are equally suited to the task of caring for others. Women are often socially conditioned to care for others from a young age. Many women are also forced to abandon their careers to care for children and the elderly.

S: Women should do all the household chores.
R: People of all genders are equally capable of doing house chores. Men are often conditioned to believe that only women do household chores.

S: Wives should take care of their husband’s parents.
R: The responsibility of taking care of elderly individuals in the family falls equally on individuals of all genders. This is not the sole remit of women.

S: Women who work outside of the home do not care about their children.
R: Working outside of the home has no correlation with a woman’s love or concern for her children. Parents of all genders may work outside of the home while also caring for their children.

S: Women who are also mothers are less competent in the office because they are distracted by childcare.
R: Women who have “double duty”, i.e., work outside the home and raise children are not less competent in the workplace.

S: Women who do not work outside the home do not contribute to the household or contribute very little in comparison to their husbands. 
R: Women who are homemakers perform unpaid domestic labour (such as cooking, cleaning, washing, household management and accounts) and care work (such as caring for the elderly and for children, helping children with their homework and extracurriculars). The unpaid labour performed by women not only contributes to the household’s quality of life but also results in monetary savings. Women who are homemakers contribute to the household to an equal (or greater) extent. Their contributions are often overlooked because men are conditioned to believe that such work is of limited value.
"""

# === Tokenizer Setup ===
ENCODER = tiktoken.get_encoding("cl100k_base")
CHUNK_SIZE = 5000


def split_into_chunks(text, max_tokens):
    tokens = ENCODER.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [ENCODER.decode(chunk) for chunk in chunks]


def rewrite_text(text: str, api_keys, models) -> (str, bool):
    chunks = split_into_chunks(text, CHUNK_SIZE)
    rewritten_chunks = []
    success = True

    for i, chunk in enumerate(chunks):
        chunk_success = False
        for model in models:
            for key_index, api_key in enumerate(api_keys):
                try:
                    print(f"Chunk {i + 1}: Trying model '{model}' with API key {key_index + 1}")
                    client = Groq(api_key=api_key)
                    message = [{"role": "user", "content": REWRITE_PROMPT + "\n\nText:\n" + chunk}]
                    response = client.chat.completions.create(model=model, messages=message)
                    rewritten_chunks.append(response.choices[0].message.content.strip())
                    time.sleep(REQUEST_DELAY)
                    chunk_success = True
                    break
                except Exception as e:
                    print(f"Error with model '{model}', key {key_index + 1}: {e}")
            if chunk_success:
                break

        if not chunk_success:
            print(f"Failed to process chunk {i + 1} with all models and keys.")
            success = False
            rewritten_chunks.append("[ERROR PROCESSING CHUNK]")

    return "\n\n".join(rewritten_chunks), success


def move_file(src_path, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    shutil.move(src_path, os.path.join(dst_folder, os.path.basename(src_path)))


def process_all_files():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(COMPLETED_FOLDER, exist_ok=True)
    os.makedirs(ERROR_FOLDER, exist_ok=True)

    files = sorted(os.listdir(INPUT_FOLDER))

    for file_name in files:
        if not file_name.endswith(".txt"):
            continue

        input_path = os.path.join(INPUT_FOLDER, file_name)
        output_path = os.path.join(OUTPUT_FOLDER, f"modified_{file_name}")

        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"\n=== Processing: {file_name} ===")
        modified_text, success = rewrite_text(text, API_KEYS, MODEL_LIST)

        if modified_text:
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(modified_text)

        if success:
            move_file(input_path, COMPLETED_FOLDER)
        else:
            move_file(input_path, ERROR_FOLDER)

        print(f"Sleeping for {FILE_DELAY // 60} minutes...")
        time.sleep(FILE_DELAY)


# === Constants ===
INPUT_FOLDER = "/home/abhisek/Thesis/Part_3/Part 4/Cases_screening50"
OUTPUT_FOLDER = "/home/abhisek/Thesis/Part_3/Part 4/modified_text_50"
COMPLETED_FOLDER = os.path.join(INPUT_FOLDER, "Completed")
ERROR_FOLDER = os.path.join(INPUT_FOLDER, "error")
REQUEST_DELAY = 10  # 1 minute
FILE_DELAY = 30    # 1 minute

if __name__ == "__main__":
    process_all_files()
