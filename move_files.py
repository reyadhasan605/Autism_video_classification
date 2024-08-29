import os
import shutil
import pandas as pd

# Load the CSV file
csv_file = '/media/zasim/69366298-8e7c-477a-a176-aba06b9848d94/reyad/autism_ds/ucf101_top5/test.csv'
df = pd.read_csv(csv_file)

# Define the base directory where your videos are located
video_base_dir = '/media/zasim/69366298-8e7c-477a-a176-aba06b9848d94/reyad/autism_ds/ucf101_top5/test'

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    video_name = row['video_name']
    tag = row['tag']

    # Define the source file path
    source = os.path.join(video_base_dir, video_name)

    # Define the destination directory path (tag folder)
    dest_dir = os.path.join(video_base_dir, tag)

    # Create the tag directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Define the destination file path
    destination = os.path.join(dest_dir, video_name)

    # Move the video file to the destination folder
    if os.path.exists(source):
        shutil.move(source, destination)
        print(f'Moved {video_name} to {dest_dir}')
    else:
        print(f'File {video_name} does not exist in the source directory.')