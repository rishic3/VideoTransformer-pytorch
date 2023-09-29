import cv2
import os
import skvideo.io
import glob
import numpy as np

def process_patient_dir(patient_dir):
    frames = sorted(glob.glob(os.path.join(input_dir, patient_dir, '*.jpg')))
    fragment_number = 0
    current_fragment_frames = []
    last_frame_number = None

    def write_fragment(fragment_frames):
        nonlocal fragment_number
        if not fragment_frames:
            return
        
        number_of_frames = len(fragment_frames)
        print(f"Starting writing fragment {fragment_number} with {number_of_frames} frames.")

        videodata = []

        for idx, frame_path in enumerate(fragment_frames):
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Error reading frame: {frame_path}")
                continue
            
            if idx == 0:  # for the first frame, set the dimension
                height, width, layers = frame.shape
            else:  # for the subsequent frames, resize them to match the first frame's dimensions
                frame = cv2.resize(frame, (width, height))
            
            videodata.append(frame)

        if len(videodata) < 1:
            print("No frames to write.")
            return
        
        output_file = os.path.join(output_dir, patient_dir, f'fragment_{fragment_number:04d}_{number_of_frames}.mp4')

        videodata = np.array(videodata, dtype=np.uint8)
        skvideo.io.vwrite(output_file, videodata)

        # Verify the number of frames in the written video
        metadata = skvideo.io.ffprobe(output_file)
        num_frames_written = int(metadata['video']['@nb_frames'])
        print(f"Number of frames in the written video: {num_frames_written}")
        
        fragment_number += 1
        print(f"Finished writing fragment {fragment_number} to {output_file}.")
    
    for frame in frames:
        frame_number = int(frame.split('_')[-1].split('.')[0])
        if last_frame_number is None or frame_number - last_frame_number <= 3:
            current_fragment_frames.append(frame)
        else:
            write_fragment(current_fragment_frames)
            current_fragment_frames = [frame]  # start a new fragment with the current frame
        last_frame_number = frame_number
    
    if current_fragment_frames:  # write the last fragment if it exists
        write_fragment(current_fragment_frames)

input_dir = "/data/datasets/rishi/cropped_frames"
output_dir = "/data/datasets/rishi/cropped_video_fragments"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for patient_dir in os.listdir(input_dir):
    patient_output_dir = os.path.join(output_dir, patient_dir)
    if not os.path.exists(patient_output_dir):
        os.makedirs(patient_output_dir)
    process_patient_dir(patient_dir)
