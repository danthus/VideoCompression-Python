import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import configparser
from util import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def compare_frames(OUTPUT_DIR,FRAME_NUMS,HEIGHT, WIDTH):

    psnr_values = []
    for frame_num in range(FRAME_NUMS):
        original_filename = f"y_frames/foreman_y_frame_{frame_num}.y"
        # print(original_filename)
        reconstructed_filename = f"{OUTPUT_DIR}/y_frames_reconstructed/foreman_y_reconstructed_frame_{frame_num}.y"
        # print(reconstructed_filename)
        with open(original_filename, 'rb') as f:
            original_frame = np.frombuffer(f.read(), dtype=np.uint8).reshape(HEIGHT, WIDTH)

        with open(reconstructed_filename, 'rb') as f:
            reconstructed_frame = np.frombuffer(f.read(), dtype=np.uint8).reshape(HEIGHT, WIDTH)

        psnr_value = psnr(original_frame, reconstructed_frame)
        # print(psnr_value)
        psnr_values.append(psnr_value)
    return psnr_values


def visualize_frames(frame_num,OUTPUT_DIR,HEIGHT, WIDTH,I):
    #global FRAME_NUMS,WIDTH, HEIGHT, I, R, N,OUTPUT_DIR  # To use these global variables within the function

    # Read original frame
    original_frame = np.fromfile(f"y_frames/foreman_y_frame_{frame_num}.y", dtype=np.uint8).reshape(HEIGHT, WIDTH)

    # Read decoded frame
    decoded_frame = read_and_pad(f"{OUTPUT_DIR}/y_frames_decoded/foreman_y_decoded_frame_{frame_num}.y", I,WIDTH,HEIGHT)[:HEIGHT, :WIDTH]
    #decoded_frame = np.fromfile(f"{OUTPUT_DIR}/y_frames_decoded/foreman_y_decoded_frame_{frame_num}.y", dtype=np.uint8).reshape(HEIGHT, WIDTH)

    # Read reconstructed frame
    reconstructed_frame = np.fromfile(f"{OUTPUT_DIR}/y_frames_reconstructed/foreman_y_reconstructed_frame_{frame_num}.y", dtype=np.uint8).reshape(HEIGHT, WIDTH)

    # Plot frames
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_frame, cmap='gray')
    axes[0].set_title('Original Frame')
    axes[0].axis('off')

    axes[1].imshow(decoded_frame, cmap='gray')
    axes[1].set_title('Decoded Frame')
    axes[1].axis('off')

    axes[2].imshow(reconstructed_frame, cmap='gray')
    axes[2].set_title('Reconstructed Frame')
    axes[2].axis('off')

    # Save the visualizations
    if not os.path.exists(f"{OUTPUT_DIR}/visualization_outputs"):
        os.makedirs(f"{OUTPUT_DIR}/visualization_outputs")
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/decoded_frame_{frame_num}.png")
    plt.close()
    #plt.tight_layout()  
    #plt.show()

def plot_mae_psnr(maes,psnrs,frame_nums,OUTPUT_DIR):
    #global FRAME_NUMS,WIDTH, HEIGHT, I, R, N,OUTPUT_DIR  # To use these global variables within the function
    frame_nums = list(range(frame_nums))
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Average MAE', color='tab:blue')
    ax1.plot(frame_nums, maes, 'o-', color='tab:blue', label='Average MAE')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim([0, 50])


    ax2 = ax1.twinx()
    ax2.set_ylabel('PSNR', color='tab:red')
    ax2.plot(frame_nums, psnrs, 's-', color='tab:red', label='PSNR')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim([0, 50])

    # Save the visualizations
    if not os.path.exists(f"{OUTPUT_DIR}/visualization_outputs"):
        os.makedirs(f"{OUTPUT_DIR}/visualization_outputs")
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/maes_psnrs.png")
    plt.close()
    #plt.show()


def visualize_intermediate_steps(frame_num, residuals_before_mc, residuals_after_mc, predicted_frames,I,OUTPUT_DIR,HEIGHT,WIDTH):
    #global FRAME_NUMS,WIDTH, HEIGHT, I, R, N,OUTPUT_DIR  # To use these global variables within the function
    if frame_num==0:
        return
    else:
        plt.figure(figsize=(20, 12))
        
        # Original frame
        original_frame = read_and_pad(f"y_frames/foreman_y_frame_{frame_num}.y", I,WIDTH,HEIGHT)[:HEIGHT, :WIDTH]
        plt.subplot(2, 3, 1)
        plt.imshow(original_frame, cmap='gray')
        plt.title('Source frame (to encode)')
        
        # Predicted frame before reconstruction
        plt.subplot(2, 3, 2)
        plt.imshow(predicted_frames[:HEIGHT, :WIDTH], cmap='gray')
        plt.title('Predicted frame with motion compensation')
        
        # Reconstructed frame
        #reconstructed_frame = read_and_pad(f"{OUTPUT_DIR}/y_frames_reconstructed/foreman_y_reconstructed_frame_{frame_num}.y", I,WIDTH,HEIGHT)[:HEIGHT, :WIDTH]
        reconstructed_frame = read_and_pad(f"y_frames/foreman_y_frame_{frame_num-1}.y", I,WIDTH,HEIGHT)[:HEIGHT, :WIDTH]
        plt.subplot(2, 3, 3)
        plt.imshow(reconstructed_frame, cmap='gray')
        plt.title('Reference (previous) frame')
        
        # Residuals before motion compensation
        plt.subplot(2, 3, 4)
        plt.imshow(residuals_before_mc[:HEIGHT, :WIDTH], cmap='gray', vmin=0, vmax=255)  # Adjusted color scale for visual clarity
        plt.title('Absolute Residuals Without MC')
        
        # Residuals after motion compensation
        plt.subplot(2, 3, 5)
        plt.imshow(residuals_after_mc[:HEIGHT, :WIDTH], cmap='gray', vmin=0, vmax=255)  # Adjusted color scale for visual clarity
        plt.title('Absolute Residuals With MC')
        
        # Save the visualizations
        if not os.path.exists(f"{OUTPUT_DIR}/visualization_outputs"):
            os.makedirs(f"{OUTPUT_DIR}/visualization_outputs")
        plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/visualization_frame_{frame_num}.png")
        plt.close()
        #plt.show()
        plt.close()

def visualize_reconstructed_frame(file_name, frame_num, I, OUTPUT_DIR, HEIGHT, WIDTH):
    plt.figure(figsize=(19, 10))
    
    original_frame = read_and_pad(f"{file_name}_y_frames/y_frame_{frame_num}.y", I,WIDTH,HEIGHT)[:HEIGHT, :WIDTH]
    plt.subplot(1, 2, 1)
    plt.imshow(original_frame, cmap='gray')
    plt.title('Source frame (to encode)')
    
    reconstructed_frame = read_and_pad(f"{OUTPUT_DIR}/y_frames_reconstructed/y_reconstructed_frame_{frame_num}.y", I,WIDTH,HEIGHT)[:HEIGHT, :WIDTH]
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_frame, cmap='gray')
    plt.title('Reconstructed frame')
    
    if not os.path.exists(f"{OUTPUT_DIR}/visualization_outputs"):
        os.makedirs(f"{OUTPUT_DIR}/visualization_outputs")
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/visualization_reconstructed_frame_{frame_num}.png")
    plt.close()


def visualize_decoded_frame(file_name, frame_num, I, OUTPUT_DIR, HEIGHT, WIDTH):
    plt.figure(figsize=(19, 10))
    
    original_frame = read_and_pad(f"{file_name}_y_frames/y_frame_{frame_num}.y", I,WIDTH,HEIGHT)[:HEIGHT, :WIDTH]
    plt.subplot(1, 2, 1)
    plt.imshow(original_frame, cmap='gray')
    plt.title('Source frame (to encode)')
    
    reconstructed_frame = read_and_pad(f"{OUTPUT_DIR}/y_frames_decoded/y_decoded_frame_{frame_num}.y", I,WIDTH,HEIGHT)[:HEIGHT, :WIDTH]
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_frame, cmap='gray')
    plt.title('Decoded frame')
    
    if not os.path.exists(f"{OUTPUT_DIR}/visualization_outputs"):
        os.makedirs(f"{OUTPUT_DIR}/visualization_outputs")
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/visualization_decoded_frame_{frame_num}.png")
    plt.close()
    
def visualize_color_inter(decoded_frame, mv_dir, I, OUTPUT_DIR):
    color_map = {
    0: (1, 0, 0, 0.5),  # Red with 50% transparency
    1: (0, 0, 1, 0.5),  # Blue with 50% transparency
    2: (0, 1, 0, 0.5),  # Green with 50% transparency
    3: (0.5, 0, 0.5, 0.5)  # Purple with 50% transparency
    }
    def load_mv_list(filename):
        mv_list = []
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    x, y, mv_x, mv_y, ref_idx = map(int, parts)
                    mv_list.append((x, y, mv_x, mv_y, ref_idx))
        return mv_list
    mv_list = load_mv_list(mv_dir)

    # Plot the frame with different cover colors with the given ref_frame_idx
    plt.imshow(decoded_frame, cmap='gray')  # Display the decoded frame in grayscale
    ax = plt.gca()  # Get current axes
    half_I = int(I / 2)
    index = 0
    while index < len(mv_list):
        if (index < len(mv_list) - 1) and (mv_list[index+1][0] - mv_list[index][0] == half_I):   # Split mode
            color_rect0 = patches.Rectangle((mv_list[index][0], mv_list[index][1]), half_I, half_I, linewidth=1, edgecolor='none', facecolor=color_map[mv_list[index][4]])
            index += 1
            ax.add_patch(color_rect0)
            color_rect1 = patches.Rectangle((mv_list[index][0], mv_list[index][1]), half_I, half_I, linewidth=1, edgecolor='none', facecolor=color_map[mv_list[index][4]])
            index += 1
            ax.add_patch(color_rect1)
            color_rect2 = patches.Rectangle((mv_list[index][0], mv_list[index][1]), half_I, half_I, linewidth=1, edgecolor='none', facecolor=color_map[mv_list[index][4]])
            index += 1
            ax.add_patch(color_rect2)
            color_rect3 = patches.Rectangle((mv_list[index][0], mv_list[index][1]), half_I, half_I, linewidth=1, edgecolor='none', facecolor=color_map[mv_list[index][4]])
            index += 1
            ax.add_patch(color_rect3)
        else:
            color_rect = patches.Rectangle((mv_list[index][0], mv_list[index][1]), I, I, linewidth=1, edgecolor='none', facecolor=color_map[mv_list[index][4]])
            ax.add_patch(color_rect)
            index += 1
    # plt.show()
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/color_inter.png")
    plt.close()

def visualize_ref_frame_and_mv_frame(decoded_frame, mv_dir, I, OUTPUT_DIR):
    color_map = {
    0: (1, 0, 0, 0.5),  # Red with 50% transparency
    1: (0, 0, 1, 0.5),  # Blue with 50% transparency
    2: (0, 1, 0, 0.5),  # Green with 50% transparency
    3: (0.5, 0, 0.5, 0.5)  # Purple with 50% transparency
    }
    def load_mv_list(filename):
        mv_list = []
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    x, y, mv_x, mv_y, ref_idx = map(int, parts)
                    mv_list.append((x, y, mv_x, mv_y, ref_idx))
        return mv_list
    mv_list = load_mv_list(mv_dir)

    # Plot the frame with different cover colors with the given ref_frame_idx
    plt.imshow(decoded_frame, cmap='gray')  # Display the decoded frame in grayscale
    ax = plt.gca()  # Get current axes
    half_I = int(I / 2)
    index = 0
    while index < len(mv_list):
        if (index < len(mv_list) - 1) and (mv_list[index+1][0] - mv_list[index][0] == half_I):   # Split mode
            color_rect0 = patches.Rectangle((mv_list[index][0], mv_list[index][1]), half_I, half_I, linewidth=1, edgecolor='none', facecolor=color_map[mv_list[index][4]])
            index += 1
            ax.add_patch(color_rect0)
            color_rect1 = patches.Rectangle((mv_list[index][0], mv_list[index][1]), half_I, half_I, linewidth=1, edgecolor='none', facecolor=color_map[mv_list[index][4]])
            index += 1
            ax.add_patch(color_rect1)
            color_rect2 = patches.Rectangle((mv_list[index][0], mv_list[index][1]), half_I, half_I, linewidth=1, edgecolor='none', facecolor=color_map[mv_list[index][4]])
            index += 1
            ax.add_patch(color_rect2)
            color_rect3 = patches.Rectangle((mv_list[index][0], mv_list[index][1]), half_I, half_I, linewidth=1, edgecolor='none', facecolor=color_map[mv_list[index][4]])
            index += 1
            ax.add_patch(color_rect3)
        else:
            color_rect = patches.Rectangle((mv_list[index][0], mv_list[index][1]), I, I, linewidth=1, edgecolor='none', facecolor=color_map[mv_list[index][4]])
            ax.add_patch(color_rect)
            index += 1
    # plt.show()
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/color_inter.png")

    # Plot the frame with vectors.
    plt.imshow(decoded_frame, cmap='gray')
    ax = plt.gca() 
    scale_factor = 7  # Adjust this factor to make arrow more visible
    for x, y, mv_x, mv_y, _ in mv_list:
        # Plot an arrow for each motion vector
        ax.quiver(x, y, mv_x*scale_factor, mv_y*scale_factor, angles='xy', scale_units='xy', scale=2, color='black')
    # plt.show()
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/vector_inter.png")

    # Plot the grid for this intra frame (to show the blocks in the graph)
    plt.imshow(decoded_frame, cmap='gray')
    ax = plt.gca()
    index = 0
    while index < len(mv_list):
        if (index < len(mv_list) - 1) and (mv_list[index+1][0] - mv_list[index][0] == half_I):   # Split mode
            black_edge_rect0 = patches.Rectangle((mv_list[index][0], mv_list[index][1]), half_I, half_I, linewidth=1, edgecolor='black', facecolor='none')
            index += 1
            ax.add_patch(black_edge_rect0)
            black_edge_rect1 = patches.Rectangle((mv_list[index][0], mv_list[index][1]), half_I, half_I, linewidth=1, edgecolor='black', facecolor='none')
            index += 1
            ax.add_patch(black_edge_rect1)
            black_edge_rect2 = patches.Rectangle((mv_list[index][0], mv_list[index][1]), half_I, half_I, linewidth=1, edgecolor='black', facecolor='none')
            index += 1
            ax.add_patch(black_edge_rect2)
            black_edge_rect3 = patches.Rectangle((mv_list[index][0], mv_list[index][1]), half_I, half_I, linewidth=1, edgecolor='black', facecolor='none')
            index += 1
            ax.add_patch(black_edge_rect3)
        else:
            black_edge_rect = patches.Rectangle((mv_list[index][0], mv_list[index][1]), I, I, linewidth=1, edgecolor='black', facecolor='none')
            index += 1
            ax.add_patch(black_edge_rect)
    # plt.show()
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/grid_inter.png")
    plt.close()


def visualize_i_mode_frame(decoded_frame, mode_dir, I, OUTPUT_DIR):
    def load_mode_list(filename):
        mode_list = []
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    x, y, mode = map(int, parts)
                    mode_list.append((x, y,mode))
        return mode_list
    mode_list = load_mode_list(mode_dir)
    
    plt.imshow(decoded_frame, cmap='gray')  # Display the decoded frame in grayscale
    ax = plt.gca()  # Get current axes
    half_I = int(I / 2)
    index = 0
    while index < len(mode_list):
        if (index < len(mode_list) - 1) and (mode_list[index+1][0] - mode_list[index][0] == half_I):   # Split mode
            if mode_list[index][2] == 0:  # Mode 0: Arrow pointing right
                ax.arrow(mode_list[index][0] + half_I // 2, mode_list[index][1] + half_I // 2, half_I // 2, 0, head_width = half_I // 4,  fc='r', ec='r')
            if mode_list[index][2] == 1:  # Mode 1: Arrow pointing down
                ax.arrow(mode_list[index][0] + half_I // 2, mode_list[index][1] + half_I // 2, 0, half_I // 2, head_width = half_I // 4,  fc='b', ec='b')
            index += 1
            if mode_list[index][2] == 0:  # Mode 0: Arrow pointing right
                ax.arrow(mode_list[index][0] + half_I // 2, mode_list[index][1] + half_I // 2, half_I // 2, 0, head_width = half_I // 4,  fc='r', ec='r')
            if mode_list[index][2] == 1:  # Mode 1: Arrow pointing down
                ax.arrow(mode_list[index][0] + half_I // 2, mode_list[index][1] + half_I // 2, 0, half_I // 2, head_width = half_I // 4,  fc='b', ec='b')
            index += 1
            if mode_list[index][2] == 0:  # Mode 0: Arrow pointing right
                ax.arrow(mode_list[index][0] + half_I // 2, mode_list[index][1] + half_I // 2, half_I // 2, 0, head_width = half_I // 4,  fc='r', ec='r')
            if mode_list[index][2] == 1:  # Mode 1: Arrow pointing down
                ax.arrow(mode_list[index][0] + half_I // 2, mode_list[index][1] + half_I // 2, 0, half_I // 2, head_width = half_I // 4,  fc='b', ec='b')
            index += 1
            if mode_list[index][2] == 0:  # Mode 0: Arrow pointing right
                ax.arrow(mode_list[index][0] + half_I // 2, mode_list[index][1] + half_I // 2, half_I // 2, 0, head_width = half_I // 4,  fc='r', ec='r')
            if mode_list[index][2] == 1:  # Mode 1: Arrow pointing down
                ax.arrow(mode_list[index][0] + half_I // 2, mode_list[index][1] + half_I // 2, 0, half_I // 2, head_width = half_I // 4,  fc='b', ec='b')
            index += 1
        else:
            if mode_list[index][2] == 0:  # Mode 0: Arrow pointing right
                ax.arrow(mode_list[index][0] + I // 2, mode_list[index][1] + I // 2, I // 2, 0, head_width = I // 4,  fc='r', ec='r')
            if mode_list[index][2] == 1:  # Mode 1: Arrow pointing down
                ax.arrow(mode_list[index][0] + I // 2, mode_list[index][1] + I // 2, 0, I // 2, head_width = I // 4,  fc='b', ec='b')
            index += 1
    '''
    for x, y, mode in mode_list:
        center_x = x + I // 2
        center_y = y + I // 2
        if mode == 0:  # Mode 0: Arrow pointing right
            ax.arrow(center_x, center_y, I // 4, 0, head_width=I // 8, head_length=I // 8, fc='r', ec='r')
        elif mode == 1:  # Mode 1: Arrow pointing down
            ax.arrow(center_x, center_y, 0, I // 4, head_width=I // 8, head_length=I // 8, fc='b', ec='b')
    '''
    ax.set_aspect('equal', adjustable='box')
    # Set limits to match the image size
    ax.set_xlim(0, decoded_frame.shape[1])
    ax.set_ylim(decoded_frame.shape[0], 0)  # Inverted to match the image coordinate system
    # plt.show()
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/vector_intra.png")

    # Plot the grid for this intra frame (to show the blocks in the graph)
    plt.imshow(decoded_frame, cmap='gray')
    ax = plt.gca()
    index = 0
    while index < len(mode_list):
        if (index < len(mode_list) - 1) and (mode_list[index+1][0] - mode_list[index][0] == half_I):   # Split mode
            black_edge_rect0 = patches.Rectangle((mode_list[index][0], mode_list[index][1]), half_I, half_I, linewidth=1, edgecolor='black', facecolor='none')
            index += 1
            ax.add_patch(black_edge_rect0)
            black_edge_rect1 = patches.Rectangle((mode_list[index][0], mode_list[index][1]), half_I, half_I, linewidth=1, edgecolor='black', facecolor='none')
            index += 1
            ax.add_patch(black_edge_rect1)
            black_edge_rect2 = patches.Rectangle((mode_list[index][0], mode_list[index][1]), half_I, half_I, linewidth=1, edgecolor='black', facecolor='none')
            index += 1
            ax.add_patch(black_edge_rect2)
            black_edge_rect3 = patches.Rectangle((mode_list[index][0], mode_list[index][1]), half_I, half_I, linewidth=1, edgecolor='black', facecolor='none')
            index += 1
            ax.add_patch(black_edge_rect3)
        else:
            black_edge_rect = patches.Rectangle((mode_list[index][0], mode_list[index][1]), I, I, linewidth=1, edgecolor='black', facecolor='none')
            index += 1
            ax.add_patch(black_edge_rect)
    # plt.show()
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/grid_intra.png")
    plt.close()
    
    
if __name__ == "__main__":
    #test assignment 2, can be commented
    def decode_header(OUTPUT_DIR):
        with open(f"{OUTPUT_DIR}/encoded_binary/header", 'rb') as f:
            header = golomb_decoding(Bits(f))
        return header
    
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read('config.txt')
    OUTPUT_DIR = config['DECODER']['output_dir']

    header = decode_header(OUTPUT_DIR)
    WIDTH = header[0]
    HEIGHT = header[1]
    I = header[2]
    QP = header[3]
    FRAME_NUMS = header[4]
    nRefFrames = header[5]
    FMEEnable = header[7]
    
    frame_num = 4
    decoded_frame = read_and_pad(f"{OUTPUT_DIR}/y_frames_decoded/foreman_y_decoded_frame_{frame_num}.y", I,WIDTH,HEIGHT)[:HEIGHT, :WIDTH]
    mv_dir = f"{OUTPUT_DIR}/mode_and_motion_vector/frame_{frame_num}_motion_vector.txt"
    visualize_color_inter(decoded_frame, mv_dir, I, OUTPUT_DIR)
    # visualize_ref_frame_and_mv_frame(decoded_frame, mv_dir, I, OUTPUT_DIR)
    
    # frame_num = 0
    # decoded_frame = read_and_pad(f"{OUTPUT_DIR}/y_frames_decoded/foreman_y_decoded_frame_{frame_num}.y", I,WIDTH,HEIGHT)[:HEIGHT, :WIDTH]
    # mode_dir = f"{OUTPUT_DIR}/mode_and_motion_vector/frame_{frame_num}_mode.txt"
    # visualize_i_mode_frame(decoded_frame, mode_dir, I, OUTPUT_DIR)
