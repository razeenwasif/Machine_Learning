import os 
import lzma 
from tqdm import tqdm 

folder_path = "./dataset/openwebtext"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"
output_path = "./dataset/"

# Ensure paths exist
os.makedirs(output_path, exist_ok=True)
os.makedirs(folder_path, exist_ok=True)

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files 

files = xz_files_in_dir(folder_path)
total_files = len(files)

# calculate the split indices 
split_index = int(total_files * 0.9) # 90% for train 
files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()

# path for training output files 
output_train_filepath = os.path.join(output_path, output_file_train)
print(f"Writing training data to: {output_train_filepath}")
# ------------------------------------------------------------------
# process the training files 
with open(output_train_filepath, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train), desc="Processing training files"): 
        file_path = os.path.join(folder_path, filename) # input dir 
        try:
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)
        except Exception as e:
            print(f"\nError processing file {filename}: {e}")

# path for validation output files 
output_val_filepath = os.path.join(output_path, output_file_val)
print(f"Writing validation data to: {output_val_filepath}")
# ------------------------------------------------------------------
# process the validation files 
with open(output_val_filepath, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val), desc="Processing validation files"):
        file_path = os.path.join(folder_path, filename)
        try:
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)
        except Exception as e:
            print(f"\nError processing file {filename}: {e}")

# write the vocabulary to txt file 
vocab_filepath = os.path.join(output_path, vocab_file)
print(f"Writing vocabulary to: {vocab_filepath}")

# sort the vocab for consistent order 
sorted_vocab = sorted(list(vocab))

with open(vocab_filepath, "w", encoding="utf-8") as vfile:
    for char in sorted_vocab:
        vfile.write(char + '\n')

print("Processing complete")
