import os
from tqdm import tqdm

# directory with all the data files
input_directory = "./writingPrompts/"
# there are three variations: train, test, valid
data = [input_directory + "train", input_directory + "test", input_directory + "valid"]

output_directory = './processed_data/'
os.makedirs(output_directory, exist_ok=False)
target_data = [output_directory + "train", output_directory + "test", output_directory + "valid"]

word_len = 800

for name_id in tqdm(range(len(data))):
    src_file = data[name_id] + ".wp_source"
    tgt_file = data[name_id] + ".wp_target"

    with open(src_file) as sf:
        prompts = sf.readlines()
    
    with open(tgt_file) as tf:
        stories = tf.readlines()

    if len(prompts) != len(stories):
        raise AssertionError("number of prompts don't align with number of stories")
    
    out_stories = []
    for i in range(len(prompts)):
        prompt = prompts[i].strip()
        story = stories[i].split()
        truncated_story = story[:word_len]
        combined = prompt + " <endprompts> " + " ".join(truncated_story)
        out_stories.append(combined)

    output_filename = target_data[name_id] + ".wp_combined"
    with open(output_filename, "w") as o:
        for line in out_stories:
            o.write(line.strip() + "\n")

    print('finished writing', output_filename)
