"""
    Summary:
        Extracts the bitrate information from all videos in a directory 
        and saves it as CSV files.
    
    Main Contributors:
     - Felix Gan               <ganf@acm.org>
     - Yunan Ding              <yunan.ding@hkust.edu.cn>

"""
from os import listdir
from os.path import (
    join, 
    dirname, realpath, basename, 
    isfile, 
    splitext)
from ffmpeg_bitrate_stats.__main__ import extract_bitrate_python_interface
from tqdm import tqdm

def extract_cover(input_file_path, _output_dir):
    # checking if it is a file
    if isfile(input_file_path):
        extract_bitrate_python_interface(_input=input_file_path, 
                                         custom_output_dir=_output_dir,
                                         aggregation="time",
                                         output_format="csv")
        # parse output folder and return exact path
        out_file_name = splitext(basename(input_file_path))[0]
        output_file_path = f"{_output_dir}/{out_file_name}.csv"
        return output_file_path
    else:
        raise FileNotFoundError
        
def main():
    cur_directory = dirname(realpath(__file__))
    
    input_dir = join(cur_directory, "1-input_videos/")
    output_dir = join(cur_directory, "2-covers/")
    video_files = [join(input_dir, f) for f in listdir(input_dir) 
                   if isfile(join(input_dir, f)) 
                   and f.endswith(".mp4")]

    for vid in tqdm(video_files):
        extract_cover(vid, output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()
    
