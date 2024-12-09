import os
import csv
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import configparser
import threading
from tqdm import tqdm

from extract_y_components import extract_and_save_y_component
from produce_BR_table import produce_BR_table
from encoder import *
from decoder import *
from util import *
from visualization import *

def decoder_main():
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read('config.txt')
    OUTPUT_DIR = config['DECODER']['output_dir']
    y_frames_path = config['DECODER']['file_name']
    header = decode_header(OUTPUT_DIR)
    WIDTH = header[0]
    HEIGHT = header[1]
    I = header[2]
    QP = header[3]
    FRAME_NUMS = header[4]
    nRefFrames = header[5]
    VBSEnable = header[6]
    FMEEnable = header[7]
    RCflag = header[8]
    ParallelMode = header[9]

    QTCs = create_all_QTC(I)
    QTCs_sub = create_all_QTC(I//2)

    x_idx, y_idx = create_reverse_diagonal_idx(I)
    xsub_idx, ysub_idx = create_reverse_diagonal_idx(I//2)

    decoding_times = []

    # Initialize the list of reference frames
    if(FMEEnable):
        reference_frames = [np.full((2*HEIGHT-2*I+1, 2*WIDTH-2*I+1, I, I), 128, dtype=np.uint8) for _ in range(nRefFrames)]
    else:
        reference_frames = [np.full((HEIGHT, WIDTH), 128, dtype=np.uint8) for _ in range(nRefFrames)]

    print(f'Decoding: WIDTH={WIDTH}, Height={HEIGHT}, I={I}, QP={QP}, Frames={FRAME_NUMS}')

    for frame_num in tqdm(range(FRAME_NUMS)):
        decoding_time, reference_frames = decode_frame(frame_num, WIDTH, HEIGHT, I, QP, QTCs, QTCs_sub, x_idx, y_idx, xsub_idx, ysub_idx, OUTPUT_DIR, reference_frames, VBSEnable, FMEEnable, nRefFrames, RCflag, ParallelMode)
        decoding_times.append(decoding_time)

    for frame_num in range(FRAME_NUMS):
        visualize_decoded_frame(y_frames_path, frame_num, I, OUTPUT_DIR, HEIGHT, WIDTH)
    
    #psnrs vs bit_size plot
    plt.clf()
    psnrs = compare_decoded_frames(y_frames_path, OUTPUT_DIR, FRAME_NUMS, HEIGHT, WIDTH)
    bit_sizes = get_bit_sizes(OUTPUT_DIR, FRAME_NUMS)
    # for psnr in psnrs:
    #     print(psnr)

    # Save decoding times to a CSV file
    with open(f'{OUTPUT_DIR}/decoding_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame", "Decoding Time"])
        for i, time in enumerate(decoding_times):
            writer.writerow([i, time])

    # Save psnrs and bit_sizes for each parameter settings
    with open(f'{OUTPUT_DIR}/RD.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["psnr", "bit_size"])
        for i in range(len(psnrs)):
            writer.writerow([psnrs[i], bit_sizes[i]])

    plt.scatter(bit_sizes, psnrs, label='PSNRs vs Bit Size', color='blue', marker='o')
    plt.xlabel('Bit Size')
    plt.ylabel('PSNR')
    plt.ylim(0, 50)
    plt.title('PSNRs vs Bit Size')

    # Save the visualizations for psnr vs bit_size.
    if not os.path.exists(f"{OUTPUT_DIR}/visualization_outputs"):
        os.makedirs(f"{OUTPUT_DIR}/visualization_outputs")
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/tradeoff.png")

    # psnrs vs frames plot
    plt.clf()
    plt.close()
    frame_nums = list(range(FRAME_NUMS))
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('PSNR', color='tab:red')
    ax1.set_xlabel('Frame')
    ax1.plot(frame_nums, psnrs, 's-', color='tab:red', label='PSNR')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_ylim([0,50])

    # Save the visualizations for psnr vs frames
    if not os.path.exists(f"{OUTPUT_DIR}/visualization_outputs"):
        os.makedirs(f"{OUTPUT_DIR}/visualization_outputs")
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/psnr_frames.png")

    #bit_sizes vs frames plot
    plt.clf()
    plt.close()
    frame_nums = list(range(FRAME_NUMS))
    fig, ax2 = plt.subplots()
    ax2.set_ylabel('bit_sizes', color='tab:red')
    ax2.set_ylim([0,300000])
    ax2.plot(frame_nums, bit_sizes, 's-', color='tab:red', label='bit_size')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Save the visualizations for bit_sizes vs frames plot
    if not os.path.exists(f"{OUTPUT_DIR}/visualization_outputs"):
        os.makedirs(f"{OUTPUT_DIR}/visualization_outputs")
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/frame_bitsize.png")

    # compression ratio vs frames plot
    plt.clf()
    plt.close()
    # compression_ratio = []
    # for frame_num in range(FRAME_NUMS):
    #     encode_binary_path = f"y_frames/foreman_y_frame_{frame_num}.y"
    #     compression_ratio.append(os.path.getsize(encode_binary_path) * 8 / bit_sizes[frame_num])

    # fig, ax3 = plt.subplots()
    # ax3.set_ylabel('compression_ratio', color='tab:red')
    # ax3.plot(frame_nums, compression_ratio, 's-', color='tab:red', label='bit_size')
    # ax3.tick_params(axis='y', labelcolor='tab:red')

    # Save the visualizations for bit_sizes vs frames plot
    if not os.path.exists(f"{OUTPUT_DIR}/visualization_outputs"):
        os.makedirs(f"{OUTPUT_DIR}/visualization_outputs")
    plt.savefig(f"{OUTPUT_DIR}/visualization_outputs/compression_vs_frameNumbers.png")
    
    print ("average psnrs:")
    print (np.mean(psnrs))

def encoder_main():
    # Read config file information
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read('config.txt')
    file_name = config['ENCODER']['file_name']
    frame_nums = int(config['ENCODER']['num_frames'])
    width = int(config['ENCODER']['width'])
    height = int(config['ENCODER']['height'])
    I = int(config['ENCODER']['i'])
    R = int(config['ENCODER']['r'])
    QP = int(config['ENCODER']['QP'])
    I_period = int(config['ENCODER']['i_period'])
    nRefFrames = int(config['ENCODER']['nRefFrames'])  # Number of reference frames
    VBSEnable = config['ENCODER'].getboolean('VBSEnable')
    const = float(config['ENCODER']['const'])
    const2 = float(config['ENCODER']['const2'])
    FMEEnable = config['ENCODER'].getboolean('FMEEnable')
    FastME = config['ENCODER'].getboolean('FastME')
    RCflag = int(config['ENCODER']['RCflag'])
    targetBR = int(config['ENCODER']['targetBR'])
    FPS = int(config['ENCODER']['FPS'])
    ParallelMode = int(config['ENCODER']['ParallelMode'])
    table_frame_nums = int(config['BRTABLE']['num_frames'])
    output_dir = f"{file_name}{frame_nums}I{I}R{R}QP{QP}-nref{nRefFrames}{VBSEnable}intra{const}inter{const2}{FMEEnable}{FastME}RC{RCflag}parallel{ParallelMode}"

    # Extract the y components and produce the related files.
    extract_nums = frame_nums
    if table_frame_nums > frame_nums:
        extract_nums = table_frame_nums
    extract_and_save_y_component(file_name, f'{file_name}_y_frames', width, height, extract_nums)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    QTCs = create_all_QTC(I)
    QTC_subs = create_all_QTC(I//2)
    Lagrange = const * 2**((QP-7)/3)
    Lagrange2 = const2 * 2**((QP-7)/3)
    # RC Enable, then produce the average row BR table.
    if not (RCflag == 0):
        produce_BR_table(file_name, table_frame_nums, width, height, I, R, QTCs, QTC_subs, nRefFrames, VBSEnable, Lagrange, Lagrange2, FMEEnable, FastME, ParallelMode, output_dir)
    
    # encode header information
    binary = golomb_encoding([width, height, I, QP, frame_nums, nRefFrames, VBSEnable, FMEEnable, RCflag, ParallelMode])
    if not os.path.exists(f'{output_dir}/encoded_binary'):
        os.makedirs(f'{output_dir}/encoded_binary')
    with open(f'{output_dir}/encoded_binary/header', 'wb') as f:
        binary.tofile(f)

    if ParallelMode == 3:
        maes, binary_lengths, encoding_time, VBS_percentages = parallel_frame_encode(f'{file_name}_y_frames', frame_nums, table_frame_nums, width, height, I, R, QP, QTCs, QTC_subs, I_period, output_dir, nRefFrames, VBSEnable, FMEEnable, FastME, Lagrange, Lagrange2, RCflag, targetBR//FPS, ParallelMode)
        print(f"total encoding time: {encoding_time} s")

    encoding_times = []
    if not (ParallelMode == 3):
        # Initialize the list of reference frames with dummy frames
        reference_frames = []
        if FMEEnable:
            reference_frames = [np.full((2*height-2*I+1, 2*width-2*I+1, I, I), 128, dtype=np.uint8) for _ in range(nRefFrames)]
            R *= 2
        if not FMEEnable:
            reference_frames = [np.full((height, width), 128, dtype=np.uint8) for _ in range(nRefFrames)]
        maes = []
        binary_lengths = []
        
        VBS_percentages = []
        print("Encoding frames")
        for frame_num in tqdm(range(frame_nums)):
            average_mae, b_len, encoding_t, reference_frames, VBS_percentage, QP = encode_frame(f'{file_name}_y_frames', frame_num, table_frame_nums, width, height, I, R, QP, QTCs, QTC_subs, I_period, output_dir, reference_frames, VBSEnable, FMEEnable, FastME, Lagrange, Lagrange2, RCflag, targetBR//FPS, ParallelMode)
            maes.append(average_mae)
            binary_lengths.append(b_len)
            encoding_times.append(encoding_t)
            VBS_percentages.append(VBS_percentage)
        print(f"total encoding time: {np.sum(encoding_times)} s")

    for frame_num in range(frame_nums):
        visualize_reconstructed_frame(file_name, frame_num, I, output_dir, height, width)
    # Save VBS percentages to a CSV file
    with open(f'{output_dir}/VBS_percentages.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame", "VBS_percentages"])
        for i, VBS_percentage in enumerate(VBS_percentages):
            writer.writerow([i, VBS_percentage])

    # Save encoding times to a CSV file
    with open(f'{output_dir}/encoding_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame", "Encoding Time"])
        for i, time in enumerate(encoding_times):
            writer.writerow([i, time])

    psnrs = compare_reconstructed_frames(file_name, output_dir, frame_nums, height, width)

    # Save MAE to another CSV file
    # with open(f'{output_dir}/MAE.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["Frame", "MAE", "PSNR", "binary length"])
    #     for i, mae in enumerate(maes):
    #         writer.writerow([i, mae, psnrs[i], binary_lengths[i]])
            
    # print('sum bit:', sum(binary_lengths))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To encode, use -> python3 main.py 1. To decode, use -> python3 main.py 2')
    parser.add_argument("mode", help="Encode or decode mode, 1 or 2")
    args = parser.parse_args()

    if(args.mode == '1' or args.mode == 'Encode' or args.mode == 'encode' or args.mode == 'ENCODE'):
        time1 = time.time()
        encoder_main()
        print('Total time spent: ', time.time()-time1, ' s')
    elif(args.mode == '2' or args.mode == 'Decode' or args.mode == 'decode' or args.mode == 'DECODE'):
        time1 = time.time()
        decoder_main()
        print('Total time spent: ', time.time()-time1, ' s')
    else:
        print("----Error argument----")
        print("To encode, use -> python3 main.py 1")
        print("To decode, use -> python3 main.py 2")
        print("Make sure config.txt is set to correct parameters before running")
        print("/"*50)
    