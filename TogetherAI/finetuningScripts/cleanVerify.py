import json
import re

input_file = './combined_data.jsonl'
output_file = './cleaned_combined_data.jsonl'

# clean text
def clean_text(text):
    text = text.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^\x20-\x7E]', '', text)

# takes input JSONL file, cleans text, puts it in the output file
with open(input_file, 'r') as inputfile, open(output_file, 'w') as outputfile:
    for line in inputfile:
        try:
            json_obj = json.loads(line)
            if "text" in json_obj:
                json_obj["text"] = clean_text(json_obj["text"])
                outputfile.write(json.dumps(json_obj) + '\n')
        except json.JSONDecodeError as e:
            print(f"Invalid format at line {line}")

print(f"written to output file")

def validate_jsonl(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                assert "text" in json_obj, "Missing 'text' field"
            except json.JSONDecodeError as e:
                print(f"Invalid format at line {line}")
            except AssertionError as e:
                print(f"error: {e}")

validate_jsonl(output_file)

