import argparse
import random
import sys
import os
import tempfile
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample random lines from a text file, with optional pre-splitting."
    )
    parser.add_argument(
        "--input_file",
        help="Path to the input text file",
        required=True
    )
    parser.add_argument(
        "--output_file",
        help="Path to the output text file",
        required=True
    )
    parser.add_argument(
        "--number_of_lines",
        type=int,
        help="Number of random lines to write to the output file",
        required=True
    )
    parser.add_argument(
        "--file_split_lines",
        type=int,
        default=None,
        help="If set, split the input file into chunks of this many lines before sampling"
    )
    return parser.parse_args()


def split_input_file(input_path, split_size):
    """
    Split the input file into multiple chunk files, each with up to split_size lines.
    Returns a list of split file paths.
    """
    split_files = []
    base_dir = tempfile.mkdtemp(prefix="file_splits_")
    file_index = 1
    line_count = 0
    out_file = None

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile):
            if line_count % split_size == 0:
                file_name = split_path = os.path.join(
                    base_dir,
                    f"{os.path.basename(input_path)}.part{file_index}.txt"
                )
                if out_file:
                    out_file.close()
                
                print(f"Writing out split file {file_name}")
                out_file = open(split_path, 'w', encoding='utf-8')
                split_files.append(split_path)
                file_index += 1
            out_file.write(line)
            line_count += 1
    if out_file:
        out_file.close()

    print(f"Input file split into {len(split_files)} parts under {base_dir}")
    return split_files


def reservoir_sample(file_paths, k):
    """
    Perform reservoir sampling of k lines over multiple files (or a single file).
    Always selects uniformly at random without replacement.
    """
    reservoir = []
    count = 0

    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as infile:
            for line in tqdm(infile):
                count += 1
                if len(reservoir) < k:
                    reservoir.append(line)
                else:
                    # Replace elements with probability k/count
                    j = random.randrange(count)
                    if j < k:
                        reservoir[j] = line

    # Shuffle to ensure random order in output
    random.shuffle(reservoir)
    return reservoir, count


def main():
    args = parse_args()

    # Determine source files: either the original file or its splits
    if args.file_split_lines:
        source_files = split_input_file(args.input_file, args.file_split_lines)
    else:
        source_files = [args.input_file]

    # Reservoir sample k lines
    try:
        sampled, total = reservoir_sample(source_files, args.number_of_lines)
    except Exception as e:
        print(f"Error during sampling: {e}", file=sys.stderr)
        sys.exit(1)

    n = args.number_of_lines
    if n < 0:
        print("--number_of_lines must be non-negative", file=sys.stderr)
        sys.exit(1)

    if n > total:
        print(
            f"Requested number of lines ({n}) exceeds total lines in input ({total})",
            file=sys.stderr
        )
        sys.exit(1)

    # Write sampled lines to the output file
    try:
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            outfile.writelines(sampled)
    except Exception as e:
        print(f"Error writing to output file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {n} random lines to {args.output_file}")


if __name__ == "__main__":
    main()
