import os
import numpy as np
import csv
import threading

from util import *
from encoder import *
from tqdm import tqdm

def interpolate_BR(BR_list, QP_list, max_QP):
    new_QP_list = np.arange(max_QP+1)
    new_BR_list = np.interp(new_QP_list, QP_list, BR_list)
    if max_QP % 2 == 1: # if maxQP is odd, the [-1] element equals [-2] element
        new_BR_list[-1] /= 2 # interpolate the element with [-2] and a zero, which is divide by 2 
    return new_BR_list.astype(np.int32), new_QP_list.astype(np.int32)
        
#   Intra_BR_produce CONSUMES:
# file_dir: input file directory
# frame_num: the index of the frame (integer)
# width
# height
# I: block size (integer)
# R: search range
# QTC
# QTC_sub
# VBSEnable
# Lagrange
# ParallelMode
# BRs: []
#   Intra_BR_produce EFFECTS:
# Bit rate for each frame will be added into BRs(a list).
def intra_BR_produce(file_dir, frame_num, width, height, I, QTC, QTC_sub, VBSEnable, Lagrange, ParallelMode, BRs):
    padded_frame = read_and_pad(f'{file_dir}/y_frame_{frame_num}.y', I, width, height)
    frame_mae, mode_list, residual_list, binary, reconstructed_frame, VBS_percentage = intra_processing(padded_frame, I, QTC, QTC_sub, VBSEnable, Lagrange, ParallelMode)
    BRs.append(len(binary) / (height // I))

#   Inter_BR_produce CONSUMES:
# file_dir: input file directory
# frame_num: the index of the frame (integer)
# width
# height
# I: block size (integer)
# R: search range
# QTC
# QTC_sub
# nRefFrames: the number of reference frames to encode. (integer)
# VBSEnable
# Lagrange
# FMEEnable
# FastME
# ParallelMode
# reconstructed_frames: A list of reconstructed frames.
# BRs: []
#   Inter_BR_produce EFFECTS:
# Bit rate for each frame (but not the first) will be added into BRs(a list).
# The reconstructed_frame will be modified after reconstructing each frame.
def inter_BR_produce(file_dir, frame_num, width, height, I, R, QTC, QTC_sub, nRefFrames, VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frames, BRs):
    padded_frame = read_and_pad(f'{file_dir}/y_frame_{frame_num}.y', I, width, height)
    if frame_num == 0:
        frame_mae, mode_list, residual_list, binary, reconstructed_frame, VBS_percentage = intra_processing(padded_frame, I, QTC, QTC_sub, VBSEnable, Lagrange, ParallelMode)
        if FMEEnable:
            FME_reconstructed_frame = frame_interpolation(reconstructed_frame, I)
            reconstructed_frames[0] = FME_reconstructed_frame
        else:
            reconstructed_frames[0] = reconstructed_frame
    else:
        reference_frames = [reconstructed_frames[0] for _ in range(nRefFrames)]
        for i in range(nRefFrames):
            if frame_num - nRefFrames + i >= 0:
                reference_frames.append(reconstructed_frames[frame_num - nRefFrames + i])
                reference_frames.pop(0)
        frame_mae, mv_list, residual_list, binary, reconstructed_frame, VBS_percentage = inter_processing(padded_frame, reference_frames, I, R, QTC, QTC_sub, VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode)
        if FMEEnable:
            FME_reconstructed_frame = frame_interpolation(reconstructed_frame, I)
            reconstructed_frames[frame_num] = FME_reconstructed_frame
        else:
            reconstructed_frames[frame_num] = reconstructed_frame
        BRs.append(len(binary) / (height // I))

# Produce_BR_table consumes these parameters to produce a csv file under the output_dir/BR_tables with name BR_table{table_frame_nums}_Intra and BR_table{table_frame_nums}_Inter.
# In the table, we can save:
# "QP", 0, 1, 2, 3 ... log2(I) + 7
# "BR", a, b, c, d ... z
# IMPORTANT: If the table with given table_frame_nums exists, the function exits.
def produce_BR_table(file_name, table_frame_nums, width, height, I, R, QTCs, QTC_subs, nRefFrames, VBSEnable, Lagrange, Lagrange2, FMEEnable, FastME, ParallelMode, output_dir):
    if not os.path.exists(f'{output_dir}/BR_tables'):
        os.makedirs(f'{output_dir}/BR_tables')
    max_QP = int(np.log2(I)+7)
    # Produce Intra BR table.
    QPs = []
    Intra_BRs = []
    if not os.path.exists(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Intra.csv'):
        print("Producing Intra bit count table")
        for QP in tqdm(range(0, max_QP+1, 2)):
            QP_sub = QP if QP == 0 else QP -1
            # The BRs will be used to save the BR of each frame.
            BRs = []
            thread_list = []
            for frame_num in range(table_frame_nums):
                thread = threading.Thread(target=intra_BR_produce,
                                        args=(f'{file_name}_y_frames', frame_num, width, height, I, QTCs[QP], QTC_subs[QP_sub], VBSEnable, Lagrange, 2, BRs))
                thread.start()
                thread_list.append(thread)
            join_threads(thread_list)
            QPs.append(QP)
            Intra_BRs.append(int(np.mean(BRs)))
            
        Intra_BRs, QPs = interpolate_BR(Intra_BRs, QPs, max_QP)
        print(Intra_BRs, QPs)
    
        with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Intra.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["QP Value"])
            writer.writerow(QPs)
            writer.writerow(["Average bit-count for I frames"])
            writer.writerow(Intra_BRs)

    if not os.path.exists(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Inter.csv'):
    # Produce InterBR table.
        QPs = []
        Inter_BRs = []
        print("Producing Inter bit count table")
        for QP in tqdm(range(0, int(np.log2(I)+7), 2)):
            QP_sub = QP if QP == 0 else QP -1
            # The BRs will be used to save the BR of each frame.
            BRs = []
            if FMEEnable:
                reconstructed_frames = [np.full((2*height-2*I+1, 2*width-2*I+1, I, I), 128, dtype=np.uint8) for _ in range(table_frame_nums)]
            if not FMEEnable:
                reconstructed_frames = [np.full((height, width), 128, dtype=np.uint8) for _ in range(table_frame_nums)]
            for frame_num in range(table_frame_nums):
                inter_BR_produce(f'{file_name}_y_frames', frame_num, width, height, I, R, QTCs[QP], QTC_subs[QP_sub], nRefFrames, VBSEnable, Lagrange2, FMEEnable, FastME, ParallelMode, reconstructed_frames, BRs)

            QPs.append(QP)
            Inter_BRs.append(int(np.mean(BRs)))
            
        Inter_BRs, QPs = interpolate_BR(Inter_BRs, QPs, max_QP)
        print(Inter_BRs, QPs)
    
        with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Inter.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["QP Value"])
            writer.writerow(QPs)
            writer.writerow(["Average bit-count for P frames"])
            writer.writerow(Inter_BRs)
