import numpy as np
import os

def extract_and_save_y_component(file_name='foreman_cif', output_directory='foreman_cif_y_frames', width = 352, height=288, num_frames=30):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # makesure the width and height must be even to calculate
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1
    with open(f'{file_name}.yuv', 'rb') as f:
        for frame_idx in range(num_frames):
            Y = np.frombuffer(f.read(width * height), dtype=np.uint8).reshape((height, width))
            U = np.frombuffer(f.read(width * height // 4), dtype=np.uint8).reshape((height // 2, width // 2))
            V = np.frombuffer(f.read(width * height // 4), dtype=np.uint8).reshape((height // 2, width // 2))

            output_file = os.path.join(output_directory, f'y_frame_{frame_idx}.y')
            if not os.path.exists(output_file):
                with open(output_file, 'wb') as out:
                    out.write(Y)

extract_and_save_y_component()