import os
from tqdm import tqdm

# Define paths
DIR = "./writingPrompts/"  # Path to the directory containing your source and target files
data = [DIR + "train", DIR + "test", DIR + "valid"]

TARGET_DIR = './processed_data/'  # Directory to save processed data
os.makedirs(TARGET_DIR, exist_ok=True)
target_data = [TARGET_DIR + "train", TARGET_DIR + "test", TARGET_DIR + "valid"]

NUM_WORDS = 300  # Adjust as needed, originally 1000

for name_id in tqdm(range(len(data))):
    with open(data[name_id] + ".wp_source") as fp, open(data[name_id] + ".wp_target") as ft:
        prompts = fp.readlines()
        stories = ft.readlines()

        assert len(prompts) == len(stories)

        new_stories = [
            prompts[i].strip() + " <endprompts> " + " ".join(stories[i].split()[:NUM_WORDS])
            for i in range(len(prompts))
        ]

        with open(target_data[name_id] + ".wp_combined", "w") as o:
            for line in new_stories:
                o.write(line.strip() + "\n")
        print('Finished writing', target_data[name_id] + ".wp_combined")

