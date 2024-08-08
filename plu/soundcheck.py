import argparse
import sys
import json
from pathlib import Path
import soundfile as sf
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser(description="Filter working audio files using soundfile", formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def check_audio_file(line):
    try:
        # Parse the JSON object from the input line
        data = json.loads(line)
        file_path = Path('wav/' + data['path'])

        # Check if the file exists
        if not file_path.exists():
            return (False, f"File does not exist: {file_path}")

        # Check if the file size is larger than 100 bytes
        if file_path.stat().st_size <= 100:
            return (False, f"File too small: {file_path}")

        # Try to load the audio file
        try:
            audio_data, samplerate = sf.read(file_path)
            # File is valid, return the original JSON line
            return (True, line.strip())
        except Exception as e:
            return (False, f"Failed to load: {file_path} - Error: {e}")

    except json.JSONDecodeError as e:
        return (False, f"Invalid JSON object: {line.strip()} - Error: {e}")


def main():
    args = parser.parse_args()

    # Read the entire input file
    input_lines = sys.stdin.read().splitlines()
    
    # Determine the number of processes to use
    num_processes = min(cpu_count() // 2, len(input_lines))

    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered to get results as they are ready
        for success, result in pool.imap_unordered(check_audio_file, input_lines):
            if success:
                print(result)
            else:
                print(result, file=sys.stderr)


if __name__ == "__main__":
    main()
