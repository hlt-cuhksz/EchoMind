import os
import tqdm
import json
import argparse

class LibrispeechtrainDownloader:
    """Downloader for Librispeech dataset."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metadata = {}
    
    def download(self) -> bool:
        librispeech_dir = os.path.join(self.output_dir, "librispeech")
        if not os.path.exists(librispeech_dir):
            os.makedirs(librispeech_dir, exist_ok=True)
            try:
                original_dir = os.getcwd()
                os.chdir(librispeech_dir)

                # Download train-clean-100 dataset
                download_success = os.system("wget https://us.openslr.org/resources/12/train-clean-100.tar.gz -O train-clean-100.tar.gz")
                if download_success != 0:
                    raise RuntimeError("Failed to download train-clean-100 dataset")
                # Download train-clean-360 dataset
                download_success = os.system("wget https://us.openslr.org/resources/12/train-clean-360.tar.gz -O train-clean-360.tar.gz")
                if download_success != 0:
                    raise RuntimeError("Failed to download train-clean-360 dataset")
                # Download train-other-500 dataset
                download_success = os.system("wget https://us.openslr.org/resources/12/train-other-500.tar.gz -O train-other-500.tar.gz")
                if download_success != 0:
                    raise RuntimeError("Failed to download train-other-500 dataset")
                
                # Extract the tar.gz dataset
                extract_success = os.system("tar -xzf train-clean-100.tar.gz")
                if extract_success != 0:
                    raise RuntimeError("Failed to extract train-clean-100 dataset")
                extract_success = os.system("tar -xzf train-clean-360.tar.gz")
                if extract_success != 0:
                    raise RuntimeError("Failed to extract train-clean-360 dataset")
                extract_success = os.system("tar -xzf train-other-500.tar.gz")
                if extract_success != 0:
                    raise RuntimeError("Failed to extract train-other-500 dataset")
                    
                # Restore original directory
                os.chdir(original_dir)
                
            except Exception as e:
                print(f"Error downloading librispeech dataset: {str(e)}")
                return False
        else:
            print("librispeech dataset already downloaded")
        
        self.metadata["librispeech"] = []
        index = 0
        metadata_path = os.path.join(self.output_dir, f"librispeech.jsonl")
        if os.path.exists(metadata_path):
            print(f"Skipping librispeech dataset because it already exists")
            return True
        
        question = "Please transcribe the spoken content into written text."

        subsets = ["train-clean-100", "train-clean-360", "train-other-500"]

        index = 0
        for subset in subsets:
            subset_dir = os.path.join(self.output_dir, "librispeech/LibriSpeech", subset)
            for spk_folder in tqdm.tqdm(os.listdir(subset_dir)):
                for chapter_folder in os.listdir(os.path.join(subset_dir, spk_folder)):
                    # get all the flac files in the chapter_folder
                    flac_files = [f for f in os.listdir(os.path.join(subset_dir, spk_folder, chapter_folder)) if f.endswith(".flac")]
                    transcript_path = os.path.join(subset_dir, spk_folder, chapter_folder, f"{spk_folder}-{chapter_folder}.trans.txt")
                    transcript_dict = {}
                    with open(transcript_path, 'r', encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split(" ", 1)
                            assert len(parts) == 2, f"Invalid line: {line}"
                            flac_file = parts[0]
                            transcript = parts[1]
                            transcript_dict[flac_file] = transcript
                    for flac_file in flac_files:
                        audio_path = os.path.join(subset_dir, spk_folder, chapter_folder, flac_file)
                        transcript = transcript_dict[flac_file.split(".")[0]]
                        assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"

                        self.metadata["librispeech"].append({
                            "task_type": "understanding",
                            
                            "conversation": [
                                {
                                    "role": "user",
                                    'message_type': 'text',
                                    "content": question
                                },
                                {
                                    "role": "user",
                                    'message_type': 'audio', 
                                    'content': audio_path
                                },
                                {
                                    "role": "assistant",
                                    "message_type": "text",
                                    "content": transcript.lower()
                                }
                            ]
                        })
                        index += 1
        
        with open(metadata_path, 'w', encoding="utf-8") as f:
            for metadata in self.metadata["librispeech"]:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        print(f"Completed processing LibriSpeech dataset. Metadata saved to {metadata_path}")
        return True
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output/data/librispeech")
    args = parser.parse_args()
    downloader = LibrispeechtrainDownloader(output_dir=args.output_dir)
    downloader.download()


