import os
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import time

# --- Configuration ---
DATASET_ID = "deepmind/code_contests"
OUTPUT_DIR = os.path.join(".", "dataset_code_contests") 
TRAIN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output_train.txt")
VAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output_val.txt")
VOCAB_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vocab.txt")
CACHE_DIR = "./hf_datasets_cache" # Optional: Specify cache directory
SEPARATOR = "\n\n<|problem_separator|>\n\n" # Clear separator between problems

# --- Ensure output directory exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Download Dataset ---
print(f"Loading dataset '{DATASET_ID}'...")
start_time = time.time()
try:
    # Load all available splits
    ds = load_dataset(DATASET_ID, cache_dir=CACHE_DIR)
except Exception as e:
    print(f"\nError loading dataset: {e}")
    print("check internet connection, or disk space.")
    exit()

print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")
print("Available splits:", ds.keys())

# --- Process Splits ---
# Define which splits go into which output file
# Common practice: combine original 'validation' and 'test' into the new validation set
split_mapping = {
    'train': TRAIN_OUTPUT_FILE,
    'validation': VAL_OUTPUT_FILE,
    'test': VAL_OUTPUT_FILE, # Append test data to the validation file
}

all_chars = set()

for original_split_name, output_filepath in split_mapping.items():
    if original_split_name not in ds:
        print(f"Warning: Split '{original_split_name}' not found in dataset. Skipping.")
        continue

    print(f"\nProcessing split '{original_split_name}' -> '{output_filepath}'...")

    # Determine write mode ('w' for the first file, 'a' for appending test to val)
    write_mode = 'w'
    if original_split_name == 'test' and os.path.exists(VAL_OUTPUT_FILE):
         write_mode = 'a' # Append if validation file already exists

    dataset_split = ds[original_split_name]
    num_examples = len(dataset_split)

    with open(output_filepath, write_mode, encoding="utf-8") as outfile:
        for i in tqdm(range(num_examples), desc=f"Writing {original_split_name}"):
            try:
                example = dataset_split[i]
                combined_text = ""

                # 1. Add description
                description = example.get("description")
                if description and isinstance(description, str):
                    combined_text += description
                    all_chars.update(set(description))

                # 2. Add solutions (iterate through the list of solutions)
                solutions = example.get("solutions")
                if solutions and isinstance(solutions, dict) and 'solution' in solutions:
                     # Handle potential multiple solutions per language if structure is different
                     # Assuming 'solution' key holds a list of solution strings or dicts
                     solution_list = solutions['solution']
                     if isinstance(solution_list, list):
                          for solution_code in solution_list:
                              if solution_code and isinstance(solution_code, str):
                                  # Add separator before each solution
                                  combined_text += "\n\n<|solution|>\n\n" + solution_code
                                  all_chars.update(set(solution_code))

                # 3. Write combined text for the problem + separator
                if combined_text: # Only write if we extracted something
                     outfile.write(combined_text)
                     outfile.write(SEPARATOR) # Add separator after the whole problem entry

            except Exception as e:
                print(f"\nError processing example {i} in split '{original_split_name}': {e}")
                # Decide whether to skip or stop
                # continue

print("\nFinished writing train/validation files.")

# --- Generate Vocabulary File ---
print(f"\nGenerating vocabulary file '{VOCAB_OUTPUT_FILE}'...")
try:
    # Sort characters for consistency
    sorted_chars = sorted(list(all_chars))
    with open(VOCAB_OUTPUT_FILE, "w", encoding="utf-8") as vfile:
        for char in sorted_chars:
            vfile.write(char + '\n') # Write one character per line
    print(f"Vocabulary saved with {len(sorted_chars)} unique characters.")
except Exception as e:
    print(f"\nError writing vocabulary file: {e}")

print("\n--- Data Preparation Complete ---")
print(f"Output files generated in: {OUTPUT_DIR}")
print(f"- Training data: {TRAIN_OUTPUT_FILE}")
print(f"- Validation data: {VAL_OUTPUT_FILE}")
print(f"- Vocabulary: {VOCAB_OUTPUT_FILE}")
