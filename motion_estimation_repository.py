import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def compute_motion_level(video_directory):
    motion_levels = []

    # Retrieve all video files in the directory
    video_files = [f for f in os.listdir(video_directory) if f.endswith('.mp4')]
    
    for video_file in tqdm(video_files, desc="Processing Videos"):

        print("Processing video:", video_file)
        
        video_path = os.path.join(video_directory, video_file)
        
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()

        if not ret:
            continue

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        avg_motion_per_frame = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_motion = np.mean(magnitude)
            avg_motion_per_frame.append(avg_motion)

            prev_gray = gray

        cap.release()
        motion_levels.append({
            'video': video_file,
            'average_motion': np.mean(avg_motion_per_frame)
        })

    motion_df = pd.DataFrame(motion_levels)

    # Normalize the average motion
    max_motion = motion_df['average_motion'].max()
    motion_df['normalized_motion'] = motion_df['average_motion'] / max_motion

    # Save to CSV
    csv_path = os.path.join(video_directory, 'motion_level.csv')
    motion_df.to_csv(csv_path, index=False)
    print(".csv file saved with name: 'motion_level.csv'")

    return motion_df

# Example how to use
video_directory = '/path/to/videos'
motion_df = compute_motion_level(video_directory)


