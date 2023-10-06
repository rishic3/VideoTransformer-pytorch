import os
import csv
import re

input_csvs = ['/data/datasets/rishi/csv_store/stratified_train_seed3.csv', 
                '/data/datasets/rishi/csv_store/stratified_valid_seed3.csv', 
                '/data/datasets/rishi/csv_store/stratified_test_seed3.csv']
output_csv = ['/data/datasets/rishi/cropped_video_fragments/fragment_train_seed3.csv',
                '/data/datasets/rishi/cropped_video_fragments/fragment_valid_seed3.csv',
                '/data/datasets/rishi/cropped_video_fragments/fragment_test_seed3.csv']
min_frames = 16

for input_csv, output_csv in zip(input_csvs, output_csv):

    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        writer.writerow(['video_path', 'start', 'end', 'label', 'caption'])
        # skip header
        next(reader)
        
        for row in reader:

            video_path, _, _, label, caption = row
            
            video_dir_name = os.path.basename(os.path.splitext(video_path)[0])
            fragments_dir = os.path.join('/data/datasets/rishi/cropped_video_fragments', video_dir_name)
            
            # loop fragments
            if os.path.exists(fragments_dir):
                for fragment_file in os.listdir(fragments_dir):
                    if fragment_file.endswith('.mp4'):

                        end_index = int(re.search(r'_(\d+).mp4$', fragment_file).group(1))
                        fragment_path = os.path.join(fragments_dir, fragment_file)
                        if end_index >= 16:
                            writer.writerow([fragment_path, 0, end_index, label, caption])
