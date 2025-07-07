import argparse
import json



def parse_args():
    parser = argparse.ArgumentParser(description='convert data')
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def convert(args):
    with open(args.input_path, "r") as src_file, open(args.output_path, "w") as dst_file:
        for line in src_file:
            data = json.loads(line)

            new_data = {
                "conversations": []
            }

            for item in data["conversations"]:
                new_data["conversations"].append({
                    "from": item["from"],
                    "value": item["value"]
                })

            dst_file.write(json.dumps(new_data) + "\n")

if __name__ == "__main__":
    args = parse_args()
    convert(args)