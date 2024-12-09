import os
import numpy as np
import time
import json
import csv
import threading
from tqdm import tqdm

from decoder import *
from util import *

def intra_full_block_processing(previous_mode, frame, QTC, reconstructed_frame, x, y, I):
    current_mode, mae, residual_block, predictor = intra_find_best_predictor(frame, reconstructed_frame, x, y, I)
    trans_quantized_block = quantization(DCT(residual_block), QTC, I)
    reconstructed_block = add_with_saturation(IDCT(inverse_quantization(trans_quantized_block, QTC, I)), predictor)
    RLE_list = run_length_encoding(diagonal(trans_quantized_block))
    diff_mode = current_mode - previous_mode
    binary = golomb_encoding(np.insert(RLE_list, 0, diff_mode))
    
    return current_mode, mae, reconstructed_block, residual_block, binary
    
def intra_sub_block_processing(previous_mode, frame, QTC, reconstructed_frame, x, y, I):
    # raster order, I is sub block size
    # x, y
    binary = Bits()
    
    current_mode1, mae1, residual_block1, predictor1 = intra_find_best_predictor(frame, reconstructed_frame, x, y, I)
    trans_quantized_block1 = DCT(residual_block1)
    trans_quantized_block1 = quantization(trans_quantized_block1, QTC, I)
    RLE_list1 = run_length_encoding(diagonal(trans_quantized_block1))
    diff_mode1 = current_mode1 - previous_mode
    binary += golomb_encoding(np.insert(RLE_list1, 0, diff_mode1))
    
    reconstructed_residual_block1 = inverse_quantization(trans_quantized_block1, QTC, I)
    reconstructed_residual_block1 = IDCT(reconstructed_residual_block1)
    reconstructed_block1 = add_with_saturation(predictor1, reconstructed_residual_block1)
    reconstructed_frame[y:y+I, x:x+I] = reconstructed_block1
    
    # x+I, y
    current_mode2, mae2, residual_block2, predictor2 = intra_find_best_predictor(frame, reconstructed_frame, x+I, y, I)
    trans_quantized_block2 = DCT(residual_block2)
    trans_quantized_block2 = quantization(trans_quantized_block2, QTC, I)
    RLE_list2 = run_length_encoding(diagonal(trans_quantized_block2))
    diff_mode2 = current_mode2 - current_mode1 # Note here, diff_mode should be different
    binary += golomb_encoding(np.insert(RLE_list2, 0, diff_mode2))
    
    reconstructed_residual_block2 = inverse_quantization(trans_quantized_block2, QTC, I)
    reconstructed_residual_block2 = IDCT(reconstructed_residual_block2)
    reconstructed_block2 = add_with_saturation(predictor2, reconstructed_residual_block2)
    reconstructed_frame[y:y+I, x+I:x+I+I] = reconstructed_block2
    
    # x, y+I
    current_mode3, mae3, residual_block3, predictor3 = intra_find_best_predictor(frame, reconstructed_frame, x, y+I, I)
    trans_quantized_block3 = DCT(residual_block3)
    trans_quantized_block3 = quantization(trans_quantized_block3, QTC, I)
    RLE_list3 = run_length_encoding(diagonal(trans_quantized_block3))
    diff_mode3 = current_mode3 - current_mode2 # Note here, diff_mode should be different
    binary += golomb_encoding(np.insert(RLE_list3, 0, diff_mode3))
    
    reconstructed_residual_block3 = inverse_quantization(trans_quantized_block3, QTC, I)
    reconstructed_residual_block3 = IDCT(reconstructed_residual_block3)
    reconstructed_block3 = add_with_saturation(predictor3, reconstructed_residual_block3)
    reconstructed_frame[y+I:y+I+I, x:x+I] = reconstructed_block3
    
    # x+I, y+I
    current_mode4, mae4, residual_block4, predictor4 = intra_find_best_predictor(frame, reconstructed_frame, x+I, y+I, I)
    trans_quantized_block4 = DCT(residual_block4)
    trans_quantized_block4 = quantization(trans_quantized_block4, QTC, I)
    RLE_list4 = run_length_encoding(diagonal(trans_quantized_block4))
    diff_mode4 = current_mode4 - current_mode3 # Note here, diff_mode should be different
    binary += golomb_encoding(np.insert(RLE_list4, 0, diff_mode4))
    
    reconstructed_residual_block4 = inverse_quantization(trans_quantized_block4, QTC, I)
    reconstructed_residual_block4 = IDCT(reconstructed_residual_block4)
    reconstructed_block4 = add_with_saturation(predictor4, reconstructed_residual_block4)
    reconstructed_frame[y+I:y+I+I, x+I:x+I+I] = reconstructed_block4
    
    reconstructed_full_block = np.vstack([np.hstack([reconstructed_block1, reconstructed_block2]), np.hstack([reconstructed_block3, reconstructed_block4])])
    
    return (current_mode1, current_mode2, current_mode3, current_mode4), (mae1+mae2+mae3+mae4)/4,\
        reconstructed_full_block, (residual_block1, residual_block2, residual_block3, residual_block4), binary

# Intra_block_processing CONSUMES:
# frame: original frame (matrix of width * height)
# QTC
# reconstructed_frame: matrix of width * height
# x
# y
# I: block size
# VBSEnable
# result_lists: [[],[],[]...] list of block_nums lists
# Intra_block_processing EFFECTS:
# 1. the related result will be saved in the correct position of the result_lists.
# result_lists contains:
#     A list of 
#     [
#     0: mae: float
#     1: VBS mode: 0(not split)/1(split)
#     2: hori_verti mode: [x, y, (0(horizontal)/1(vertical))] / [[x1, y1, 0/1], [x2, y2, 0/1], [x3, y3, 0/1], [x4, y4, 0/1]](split mode)
#     3: residual block: [x, y, I * I matrix] / [[x1, y1, I/2 * I/2 matrix], [x2, y2, I/2 * I/2 matrix], [x3, y3, I/2 * I/2 matrix], [x4, y4, I/2 * I/2 matrix]]
#     4: block binary (This binary only contains the residual blocks)
#     ]
# 2. reconstructed_frame will be update.
def intra_block_processing (frame, QTC, QTC_sub, reconstructed_frame, x, y, I, VBSEnable, Lagrange, ParallelMode, result_lists):
    result_idx = (y // I) * (frame.shape[1] // I) + x // I
    previous_mode = 0
    if x > 0:
        if result_lists[result_idx-1][1] == 0: # split or not
            previous_mode = result_lists[result_idx-1][2][2] # no split
        else:
            previous_mode = result_lists[result_idx-1][2][3][2] # split
    current_mode, mae, reconstructed_block, residual_block, b = intra_full_block_processing(previous_mode, frame, QTC, reconstructed_frame, x, y, I)
    
    if VBSEnable:
        sub_current_mode, sub_mae, sub_reconstructed_block, sub_residual_block, sub_b = intra_sub_block_processing(previous_mode, frame, QTC_sub, reconstructed_frame, x, y, I//2)
        sub_b = golomb_encoding([1]) + sub_b
        SAD1 = np.sum(np.abs(frame[y:y+I, x:x+I] - reconstructed_block))
        SAD2 = np.sum(np.abs(frame[y:y+I, x:x+I] - sub_reconstructed_block))
        RDO1 = SAD1 + Lagrange * len(golomb_encoding([0]) + b)
        RDO2 = SAD2 + Lagrange * len(sub_b)
        if RDO1 > RDO2: # Split mode won.
            result_lists[result_idx].append(sub_mae)
            result_lists[result_idx].append(1)
            result_lists[result_idx].append([[x, y, sub_current_mode[0]],
                                             [x + I // 2, y, sub_current_mode[1]],
                                             [x, y + I // 2, sub_current_mode[2]],
                                             [x + I // 2, y + I // 2, sub_current_mode[3]]])
            result_lists[result_idx].append([[x, y, sub_residual_block[0]],
                                             [x + I // 2, y, sub_residual_block[1]],
                                             [x, y + I // 2, sub_residual_block[2]],
                                             [x + I // 2, y + I // 2, sub_residual_block[3]]])
            result_lists[result_idx].append(sub_b)
            return
    result_lists[result_idx].append(mae)
    result_lists[result_idx].append(0)
    result_lists[result_idx].append([x, y, current_mode])
    result_lists[result_idx].append([x, y, residual_block])
    if VBSEnable:
        b = golomb_encoding([0]) + b
    result_lists[result_idx].append(b)
    reconstructed_frame[y:y+I, x:x+I] = reconstructed_block

#   Intra_processing CONSUMES:
# frame: original frame showed in matrix with width * height
# I: block size
# QTC
# QTC_sub
# VBSEnable
# Lagrange
# ParallelMode
#   Intra_processing PRODUCES:
# frame_mae: float
# mode_list: list of [x, y, (0(horizontal)/1(vertical))]
# residual_list: list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...]]
# binary
# reconstructed_frame: reconstructed frame showed in matrix with width * height
# VBS_percentage: float
def intra_processing(frame, I, QTC, QTC_sub, VBSEnable, Lagrange, ParallelMode):
    frame_mae = 0
    mode_list = []
    residual_list = []
    reconstructed_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    binary = Bits()
    sub_count = 0
    VBS_percentage = 0.0
    row_nums = frame.shape[0] // I
    col_nums = frame.shape[1] // I
    total_count = row_nums * col_nums

    if ParallelMode == 2 or ParallelMode == 3:
        total_layer = row_nums + col_nums - 1
        result_lists = [[] for _ in range(total_count)]
        # result_lists contains:
        # A list of 
        # [
        # 0: mae: float
        # 1: VBS mode: 0(not split)/1(split)
        # 2: hori_verti mode: [x, y, (0(horizontal)/1(vertical))] / [[x1, y1, 0/1], [x2, y2, 0/1], [x3, y3, 0/1], [x4, y4, 0/1]](split mode)
        # 3: residual block: [x, y, I * I matrix] / [[x1, y1, I/2 * I/2 matrix], [x2, y2, I/2 * I/2 matrix], [x3, y3, I/2 * I/2 matrix], [x4, y4, I/2 * I/2 matrix]]
        # 4: block binary
        # ]
        for i in range(total_layer):
            thread_list = []
            for x in range(min(i + 1, frame.shape[1] // I)):
                y = i - x
                if y >= 0 and y < (frame.shape[0] // I):    # Valid y column
                    thread = threading.Thread(target=intra_block_processing,
                                              args=(frame, QTC, QTC_sub, reconstructed_frame, x * I, y * I, I, VBSEnable, Lagrange, ParallelMode, result_lists))
                    thread.start()
                    thread_list.append(thread)
            join_threads(thread_list)
        # Integrate the results.
        for result_list in result_lists:
            frame_mae += result_list[0]
            if result_list[1] == 0: # Not split mode
                mode_list.append(result_list[2])
                residual_list.append(result_list[3])
                binary += result_list[4]
            if result_list[1] == 1: # Split mode
                sub_count += 1
                mode_list.append(result_list[2][0])
                mode_list.append(result_list[2][1])
                mode_list.append(result_list[2][2])
                mode_list.append(result_list[2][3])
                residual_list.append(result_list[3][0])
                residual_list.append(result_list[3][1])
                residual_list.append(result_list[3][2])
                residual_list.append(result_list[3][3])
                binary += result_list[4]
        frame_mae /= total_count
        if VBSEnable:
            VBS_percentage = sub_count / total_count
    
    else:
        for y in range(0, frame.shape[0], I):
            previous_mode = 0
            for x in range(0, frame.shape[1], I):
                if(VBSEnable):
                    current_mode1, mae1, reconstructed_block1, residual_block1, b1 = intra_full_block_processing(previous_mode, frame, QTC, reconstructed_frame, x, y, I)
                    current_mode2, mae2, reconstructed_block2, residual_block2, b2 = intra_sub_block_processing(previous_mode, frame, QTC_sub, reconstructed_frame, x, y, I//2)
                    
                    SAD1 = np.sum(np.abs(frame[y:y+I, x:x+I] - reconstructed_block1))
                    SAD2 = np.sum(np.abs(frame[y:y+I, x:x+I] - reconstructed_block2))
                    
                    RDO1 = SAD1 + Lagrange * len(b1)
                    RDO2 = SAD2 + Lagrange * len(b2)
                    if(RDO1 > RDO2):    # Split mode
                        previous_mode = current_mode2[3]
                        binary += golomb_encoding([1]) # split
                        binary += b2
                        
                        frame_mae += mae2
                        mode_list.append((x, y, current_mode2[0]))
                        mode_list.append((x+I//2, y, current_mode2[1]))
                        mode_list.append((x, y+I//2, current_mode2[2]))
                        mode_list.append((x+I//2, y+I//2, current_mode2[3]))
                        residual_list.append((x,y, residual_block2[0]))
                        residual_list.append((x+I//2,y, residual_block2[1]))
                        residual_list.append((x,y+I//2, residual_block2[2]))
                        residual_list.append((x+I//2,y+I//2, residual_block2[3]))
                        sub_count += 1
                    else:
                        previous_mode = current_mode1
                        binary += golomb_encoding([0]) # no split
                        binary += b1
                        
                        # Note: intra_sub_block_processing overwites the reconstructed_frame, write it back
                        reconstructed_frame[y:y+I, x:x+I] = reconstructed_block1
                        
                        frame_mae += mae1
                        mode_list.append((x, y, current_mode1))
                        residual_list.append((x, y, residual_block1))
                else:
                    current_mode, mae, reconstructed_Block, residual_block, b = intra_full_block_processing(previous_mode, frame, QTC, reconstructed_frame, x, y, I)
                    previous_mode = current_mode
                    binary += b

                    reconstructed_frame[y:y+I, x:x+I] = reconstructed_Block
                    
                    frame_mae += mae
                    mode_list.append((x, y, current_mode))
                    residual_list.append((x,y, residual_block))
        frame_mae /= total_count
        if VBSEnable:
            VBS_percentage = sub_count/total_count
    
    return frame_mae, mode_list, residual_list, binary, reconstructed_frame, VBS_percentage

#   Intra_processing_one_row CONSUMES:
# original_frame: (matrix of width * height)
# reconstructed_frame: (matrix of width * height)
# y: the first index of the encoding row
# I: block size (integer)
# R: move range (integer)
# QTC
# QTC_sub
# VBSEnable
# Lagrange
# ParallelMode
# row_result_list: []
#   Intra_processing_one_row EFFECTS:
# 1. reconstructed_frame of this row will be updated.
# 2. row_result_list will be used to save the results:
# [
#   row_mae (float)
#   row_mode_list (list of [x, y, (0(horizontal)/1(vertical))]))
#   row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...]])
#   row_binary (binary file) (VBS mode and hori_verti mode are both differential, started from the first element of this row.)
#   VBS_count (integer)
# ]
# IMPORTANT: The binary file does not contain the QP value.
def intra_processing_one_row(original_frame, reconstructed_frame, y, I, QTC, QTC_sub, VBSEnable, Lagrange, ParallelMode, row_result_list):
    row_nums = original_frame.shape[0] // I
    col_nums = original_frame.shape[1] // I
    result_lists = [[] for _ in range(row_nums * col_nums)]
    # result_lists contains:
    # A list of 
    # [
    # 0: mae: float
    # 1: VBS mode: 0(not split)/1(split)
    # 2: hori_verti mode: [x, y, (0(horizontal)/1(vertical))] / [[x1, y1, 0/1], [x2, y2, 0/1], [x3, y3, 0/1], [x4, y4, 0/1]](split mode)
    # 3: residual block: [x, y, I * I matrix] / [[x1, y1, I/2 * I/2 matrix], [x2, y2, I/2 * I/2 matrix], [x3, y3, I/2 * I/2 matrix], [x4, y4, I/2 * I/2 matrix]]
    # 4: block binary
    # ]
    for x in range(col_nums):
        intra_block_processing(original_frame, QTC, QTC_sub, reconstructed_frame, x * I, y, I, VBSEnable, Lagrange, ParallelMode, result_lists)
    row_mae = 0
    row_mode_list = []
    row_residual_list = []
    row_binary = Bits()
    VBS_count = 0
    for result_list in result_lists:
        if result_list == []:
            continue
        row_mae += result_list[0]
        if result_list[1] == 0: # not split mode
            row_mode_list.append(result_list[2])
            row_residual_list.append(result_list[3])
            row_binary += result_list[4]
        else:   # split mode
            row_mode_list.append(result_list[2][0])
            row_mode_list.append(result_list[2][1])
            row_mode_list.append(result_list[2][2])
            row_mode_list.append(result_list[2][3])
            row_residual_list.append(result_list[3][0])
            row_residual_list.append(result_list[3][1])
            row_residual_list.append(result_list[3][2])
            row_residual_list.append(result_list[3][3])
            row_binary += result_list[4]
            VBS_count += 1
    row_mae /= col_nums
    row_result_list.append(row_mae)
    row_result_list.append(row_mode_list)
    row_result_list.append(row_residual_list)
    row_result_list.append(row_binary)
    row_result_list.append(VBS_count)

#   Intra_QP_processing CONSUMES:
# original_frame: (matrix of width * height)
# I: block size (integer)
# QTC
# QTC_sub
# VBSEnable
# Lagrange
# ParallelMode
# result_list: []
#   Intra_QP_processing EFFECTS:
# the result_list will be used to save the results:
# [
#   [row0_mae, row1_mae, row2_mae...], (float list)
#   [row0_mode_list, row1_mode_list, row2_mode_list...], (list of [x, y, (0(horizontal)/1(vertical))])
#   [row0_residual_list, row1_residual_list, row2_residual_list...], (list of [x, y [[a1,b1,c1,d1...], [a2,b2,c2,d2...]...] (I * I or (I//2 * I//2))])
#   [row0_binary, row1_binary, row2_binary...], (binary list) (only use differential in each row.)
#   [reconstructed_row0, reconstructed_row1, reconstructed_row2], (list of [[a1,b1,c1,d1...z1], [a2,b2,c2,d2...]...] (I * WIDTH))
#   [row0_VBS_count, row1_VBS_count, row2_VBS_count...] (integer list)
# ]
def intra_rows_processing(original_frame, I, QTC, QTC_sub, VBSEnable, Lagrange, ParallelMode, result_list):
    row_nums = original_frame.shape[0] // I
    col_nums = original_frame.shape[1] // I
    total_layer = row_nums + col_nums - 1
    rows_mae = []
    rows_mode_list = []
    rows_residual_list = []
    rows_binary = []
    rows_reconstructed = []
    rows_VBS_count = []
    reconstructed_frame = np.zeros((original_frame.shape[0], original_frame.shape[1]), dtype=np.uint8)
    if ParallelMode == 2 or ParallelMode == 3:
        result_lists = [[] for _ in range(row_nums * col_nums)]
        # result_lists contains:
        # A list of 
        # [
        # 0: mae: float
        # 1: VBS mode: 0(not split)/1(split)
        # 2: hori_verti mode: [x, y, (0(horizontal)/1(vertical))] / [[x1, y1, 0/1], [x2, y2, 0/1], [x3, y3, 0/1], [x4, y4, 0/1]](split mode)
        # 3: residual block: [x, y, I * I matrix] / [[x1, y1, I/2 * I/2 matrix], [x2, y2, I/2 * I/2 matrix], [x3, y3, I/2 * I/2 matrix], [x4, y4, I/2 * I/2 matrix]]
        # 4: block binary
        # ]
        for i in range(total_layer):
            thread_list = []
            for x in range(min(i + 1, original_frame.shape[1] // I)):
                y = i - x
                if y >= 0 and y < (original_frame.shape[0] // I):    # Valid y column
                    thread = threading.Thread(target=intra_block_processing,
                                              args=(original_frame, QTC, QTC_sub, reconstructed_frame, x * I, y * I, I, VBSEnable, Lagrange, ParallelMode, result_lists))
                    thread.start()
                    thread_list.append(thread)
            join_threads(thread_list)
        # Load data to the result_list row by row
        for row in range(row_nums):
            row_mae = 0.0
            row_mode_list = []
            row_residual_list = []
            row_binary = Bits()
            row_VBS_count = 0
            for x in range(col_nums):
                result_idx = row * col_nums + x
                block_result_list = result_lists[result_idx]
                row_mae += block_result_list[0]
                if block_result_list[1] == 0: # not split mode
                    row_mode_list.append(block_result_list[2])
                    row_residual_list.append(block_result_list[3])
                    row_binary += block_result_list[4]
                else:   # split mode
                    row_mode_list.append(block_result_list[2][0])
                    row_mode_list.append(block_result_list[2][1])
                    row_mode_list.append(block_result_list[2][2])
                    row_mode_list.append(block_result_list[2][3])
                    row_residual_list.append(block_result_list[3][0])
                    row_residual_list.append(block_result_list[3][1])
                    row_residual_list.append(block_result_list[3][2])
                    row_residual_list.append(block_result_list[3][3])
                    row_binary += block_result_list[4]
                    row_VBS_count += 1
            rows_mae.append(row_mae / col_nums)
            rows_mode_list.append(row_mode_list)
            rows_residual_list.append(row_residual_list)
            rows_binary.append(row_binary)
            rows_reconstructed.append(reconstructed_frame[row*I:(row+1)*I, :])
            rows_VBS_count.append(row_VBS_count)
    
    else:   # Sequential mode
        for row in range(row_nums):
            row_result_list = []
            # [
            #   row_mae (float)
            #   row_mode_list (list of [x, y, (0(horizontal)/1(vertical))]))
            #   row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...]])
            #   row_binary (row binary file)
            #   VBS_count (integer)
            # ]
            intra_processing_one_row(original_frame, reconstructed_frame, row * I, I, QTC, QTC_sub, VBSEnable, Lagrange, ParallelMode, row_result_list)
            rows_mae.append(row_result_list[0])
            rows_mode_list.append(row_result_list[1])
            rows_residual_list.append(row_result_list[2])
            rows_binary.append(row_result_list[3])
            rows_reconstructed.append(reconstructed_frame[row*I:(row+1)*I, :])
            rows_VBS_count.append(row_result_list[4])
    result_list.append(rows_mae)
    result_list.append(rows_mode_list)
    result_list.append(rows_residual_list)
    result_list.append(rows_binary)
    result_list.append(rows_reconstructed)
    result_list.append(rows_VBS_count)

def inter_full_block_processing(current_frame, reference_frames, previous_mv, x, y, I, R, QTC, FMEEnable, FastME):
    best_mv = (0, 0)
    best_ref_idx = 0
    lowest_mae = float('inf')
    MVP = [previous_mv[0], previous_mv[1]]
    for ref_idx, ref_frame in enumerate(reference_frames):
        if FastME and FMEEnable:
            mv, mae = find_best_block_fastME_FME(current_frame, ref_frame, x, y, I, R, current_frame.shape[1], current_frame.shape[0], MVP, 0)
        elif FastME:
            mv, mae = find_best_block_fastME(current_frame, ref_frame, x, y, I, R, current_frame.shape[1], current_frame.shape[0], MVP)
        elif FMEEnable:
            mv, mae = find_best_block_FME(current_frame, ref_frame, x, y, I, R, 0)
        else:
            mv, mae = find_best_block(current_frame, ref_frame, x, y, I, R, current_frame.shape[1], current_frame.shape[0])
        
        if mae < lowest_mae:
            lowest_mae = mae
            best_mv = mv
            best_ref_idx = ref_idx
    
    if(FMEEnable):
        predicted_block = reference_frames[best_ref_idx][y*2+best_mv[1], x*2 +best_mv[0]]
    else:
        predicted_block = reference_frames[best_ref_idx][y+best_mv[1]:y+best_mv[1]+I, x+best_mv[0]:x+best_mv[0]+I]
    
    current_block = current_frame[y:y+I, x:x+I]
    residual_block = current_block.astype(np.int16) - predicted_block.astype(np.int16)
    trans_quantized_block = quantization(DCT(residual_block), QTC, I)
    reconstructed_block = add_with_saturation(IDCT(inverse_quantization(trans_quantized_block, QTC, I)), predicted_block)
    RLE_list = run_length_encoding(diagonal(trans_quantized_block))
    diff_mv = (best_mv[0] - previous_mv[0], best_mv[1] - previous_mv[1], best_ref_idx - previous_mv[2])
    binary = golomb_encoding(np.insert(RLE_list, 0, diff_mv))
    
    return lowest_mae, best_mv, best_ref_idx, residual_block, reconstructed_block, binary

def inter_sub_block_processing(current_frame, reference_frames, previous_mv, x, y, I, R, QTC, FMEEnable, FastME):
    best_mv1 = (0, 0)
    best_mv2 = (0, 0)
    best_mv3 = (0, 0)
    best_mv4 = (0, 0)
    best_ref_idx1 = 0
    best_ref_idx2 = 0
    best_ref_idx3 = 0
    best_ref_idx4 = 0
    lowest_mae1 = float('inf')
    lowest_mae2 = float('inf')
    lowest_mae3 = float('inf')
    lowest_mae4 = float('inf')
    MVP = [previous_mv[0], previous_mv[1]]
    #x, y
    for ref_idx, ref_frame in enumerate(reference_frames):
        if FastME and FMEEnable:
            mv1, mae1 = find_best_block_fastME_FME(current_frame, ref_frame, x, y, I, R, current_frame.shape[1], current_frame.shape[0], MVP, 1)
        elif FastME:
            mv1, mae1 = find_best_block_fastME(current_frame, ref_frame, x, y, I, R, current_frame.shape[1], current_frame.shape[0], MVP)
        elif FMEEnable:
            mv1, mae1 = find_best_block_FME(current_frame, ref_frame, x, y, I, R, 1)
        else:
            mv1, mae1 = find_best_block(current_frame, ref_frame, x, y, I, R, current_frame.shape[1], current_frame.shape[0])
    
        if mae1 < lowest_mae1:
            lowest_mae1 = mae1
            best_mv1 = mv1
            best_ref_idx1 = ref_idx
    
    # x+I, y
    for ref_idx, ref_frame in enumerate(reference_frames):
        if FastME and FMEEnable:
            mv2, mae2 = find_best_block_fastME_FME(current_frame, ref_frame, x, y, I, R, current_frame.shape[1], current_frame.shape[0], best_mv1, 2)
        elif FastME:
            mv2, mae2 = find_best_block_fastME(current_frame, ref_frame, x+I, y, I, R, current_frame.shape[1], current_frame.shape[0], best_mv1)
        elif FMEEnable:
            mv2, mae2 = find_best_block_FME(current_frame, ref_frame, x, y, I, R, 2)
        else:
            mv2, mae2 = find_best_block(current_frame, ref_frame, x+I, y, I, R, current_frame.shape[1], current_frame.shape[0])
    
        if mae2 < lowest_mae2:
            lowest_mae2 = mae2
            best_mv2 = mv2
            best_ref_idx2 = ref_idx
            
    # x, y+I       
    for ref_idx, ref_frame in enumerate(reference_frames):
        if FastME and FMEEnable:
            mv3, mae3 = find_best_block_fastME_FME(current_frame, ref_frame, x, y, I, R, current_frame.shape[1], current_frame.shape[0], best_mv2, 3)
        elif FastME:
            mv3, mae3 = find_best_block_fastME(current_frame, ref_frame, x, y+I, I, R, current_frame.shape[1], current_frame.shape[0], best_mv2)
        elif FMEEnable:
            mv3, mae3 = find_best_block_FME(current_frame, ref_frame, x, y, I, R, 3)
        else:
            mv3, mae3 = find_best_block(current_frame, ref_frame, x, y+I, I, R, current_frame.shape[1], current_frame.shape[0])
    
        if mae3 < lowest_mae3:
            lowest_mae3 = mae3
            best_mv3 = mv3
            best_ref_idx3 = ref_idx
    
    # x+I, y+I       
    for ref_idx, ref_frame in enumerate(reference_frames):
        if FastME and FMEEnable:
            mv4, mae4 = find_best_block_fastME_FME(current_frame, ref_frame, x, y, I, R, current_frame.shape[1], current_frame.shape[0], best_mv3, 4)
        elif FastME:
            mv4, mae4 = find_best_block_fastME(current_frame, ref_frame, x+I, y+I, I, R, current_frame.shape[1], current_frame.shape[0], best_mv3)
        elif FMEEnable:
            mv4, mae4 = find_best_block_FME(current_frame, ref_frame, x, y, I, R, 4)
        else:
            mv4, mae4 = find_best_block(current_frame, ref_frame, x+I, y+I, I, R, current_frame.shape[1], current_frame.shape[0])
    
        if mae4 < lowest_mae4:
            lowest_mae4 = mae4
            best_mv4 = mv4
            best_ref_idx4 = ref_idx
            
    if(FMEEnable):
        predicted_block1 = reference_frames[best_ref_idx1][y*2+best_mv1[1], x*2 +best_mv1[0]][0:I, 0:I]
        predicted_block2 = reference_frames[best_ref_idx2][y*2+best_mv2[1], x*2 +best_mv2[0]][0:I, I:I+I]
        predicted_block3 = reference_frames[best_ref_idx3][y*2+best_mv3[1], x*2 +best_mv3[0]][I:I+I, 0:I]
        predicted_block4 = reference_frames[best_ref_idx4][y*2+best_mv4[1], x*2 +best_mv4[0]][I:I+I, I:I+I]
    else:
        predicted_block1 = reference_frames[best_ref_idx1][y+best_mv1[1]:y+best_mv1[1]+I, x+best_mv1[0]:x+best_mv1[0]+I]
        predicted_block2 = reference_frames[best_ref_idx2][y+best_mv2[1]:y+best_mv2[1]+I, x+best_mv2[0]+I:x+best_mv2[0]+I+I]
        predicted_block3 = reference_frames[best_ref_idx3][y+best_mv3[1]+I:y+best_mv3[1]+I+I, x+best_mv3[0]:x+best_mv3[0]+I]
        predicted_block4 = reference_frames[best_ref_idx4][y+best_mv4[1]+I:y+best_mv4[1]+I+I, x+best_mv4[0]+I:x+best_mv4[0]+I+I]
        
    current_block1 = current_frame[y:y+I, x:x+I]
    current_block2 = current_frame[y:y+I, x+I:x+I+I]
    current_block3 = current_frame[y+I:y+I+I, x:x+I]
    current_block4 = current_frame[y+I:y+I+I, x+I:x+I+I]
    residual_block1 = current_block1.astype(np.int16) - predicted_block1.astype(np.int16)
    residual_block2 = current_block2.astype(np.int16) - predicted_block2.astype(np.int16)
    residual_block3 = current_block3.astype(np.int16) - predicted_block3.astype(np.int16)
    residual_block4 = current_block4.astype(np.int16) - predicted_block4.astype(np.int16)
    trans_quantized_block1 = quantization(DCT(residual_block1), QTC, I)
    trans_quantized_block2 = quantization(DCT(residual_block2), QTC, I)
    trans_quantized_block3 = quantization(DCT(residual_block3), QTC, I)
    trans_quantized_block4 = quantization(DCT(residual_block4), QTC, I)
    RLE_list1 = run_length_encoding(diagonal(trans_quantized_block1))
    RLE_list2 = run_length_encoding(diagonal(trans_quantized_block2))
    RLE_list3 = run_length_encoding(diagonal(trans_quantized_block3))
    RLE_list4 = run_length_encoding(diagonal(trans_quantized_block4))
    
    diff_mv1 = (best_mv1[0] - previous_mv[0], best_mv1[1] - previous_mv[1], best_ref_idx1 - previous_mv[2])
    diff_mv2 = (best_mv2[0] - best_mv1[0], best_mv2[1] - best_mv1[1], best_ref_idx2 - best_ref_idx1)
    diff_mv3 = (best_mv3[0] - best_mv2[0], best_mv3[1] - best_mv2[1], best_ref_idx3 - best_ref_idx2)
    diff_mv4 = (best_mv4[0] - best_mv3[0], best_mv4[1] - best_mv3[1], best_ref_idx4 - best_ref_idx3)
    
    binary = golomb_encoding(np.insert(RLE_list1, 0, diff_mv1))
    binary += golomb_encoding(np.insert(RLE_list2, 0, diff_mv2))
    binary += golomb_encoding(np.insert(RLE_list3, 0, diff_mv3))
    binary += golomb_encoding(np.insert(RLE_list4, 0, diff_mv4))
    
    reconstructed_block1 = add_with_saturation(IDCT(inverse_quantization(trans_quantized_block1, QTC, I)), predicted_block1)
    reconstructed_block2 = add_with_saturation(IDCT(inverse_quantization(trans_quantized_block2, QTC, I)), predicted_block2)
    reconstructed_block3 = add_with_saturation(IDCT(inverse_quantization(trans_quantized_block3, QTC, I)), predicted_block3)
    reconstructed_block4 = add_with_saturation(IDCT(inverse_quantization(trans_quantized_block4, QTC, I)), predicted_block4)
    
    reconstructed_full_block = np.vstack([np.hstack([reconstructed_block1, reconstructed_block2]), np.hstack([reconstructed_block3, reconstructed_block4])])
    
    return (lowest_mae1+lowest_mae2+lowest_mae3+lowest_mae4)/4, (best_mv1, best_mv2, best_mv3, best_mv4),\
        (best_ref_idx1, best_ref_idx2, best_ref_idx3, best_ref_idx4), (residual_block1, residual_block2, residual_block3, residual_block4), reconstructed_full_block, binary

#   Inter_block_processing CONSUMES:
# current_frame: (matrix of width * height)
# reference_frames: list of (matrix of width * height)
# previous_mv: previous move vector ([dx, dy, dref])
# x
# y
# I: block size (integer)
# R: move range (integer)
# QTC
# QTC_sub
# VBSEnable
# FMEEnable
# FastME
# ParallelMode
# reconstructed_frame: (matrix of width * height)
# result_lists: [[],[],[]...] list of block_nums lists
#   Inter_block_processing EFFECTS:
# 1. the related result will be saved in the correct position of the result_lists.
# result_lists contains:
#     A list of 
#     [
#     mae: float
#     VBS_mode: 0/1 (not split/split)
#     mv_list: [x, y, dx, dy, dref](not split mode) / [[x1, y1, dx1, dy1, dref1], [x2, y2, dx2, dy2, dref2], [x3, y3, dx3, dy3, dref3], [x4, y4, dx4, dy4, dref4]](split mode)
#     residual_block: [x, y, residual block](not split mode) / [[x1, y1, residual block1], [x2, y2, residual block2], [x3, y3, residual block3], [x4, y4, residual block4]](split mode)
#     block_binary: the binary includes mv_vector. (ParallelMode == 1: mv differential disable; Else: mv differential enable) (include VBS mode and mv and residual block)
#     ]
# 2. reconstructed_frame will be update.
def inter_block_processing(current_frame, reference_frames, previous_mv, x, y, I, R, QTC, QTC_sub, VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frame, result_lists):
    result_idx = (y // I) * (current_frame.shape[1] // I) + x // I
    mae, best_mv, best_ref_idx, residual_block, reconstructed_block, b = inter_full_block_processing(current_frame, reference_frames, previous_mv, x, y, I, R, QTC, FMEEnable, FastME)
    if VBSEnable:
        sub_mae, sub_best_mv, sub_best_ref_idx, sub_residual_block, sub_reconstructed_block, sub_b = inter_sub_block_processing(current_frame, reference_frames, previous_mv, x, y, I//2, R, QTC_sub, FMEEnable, FastME)
        sub_b = golomb_encoding([1]) + sub_b
        SAD1 = np.sum(np.abs(current_frame[y:y+I, x:x+I] - reconstructed_block))
        SAD2 = np.sum(np.abs(current_frame[y:y+I, x:x+I] - sub_reconstructed_block))
        RDO1 = SAD1 + Lagrange * len(golomb_encoding([0]) + b)
        RDO2 = SAD2 + Lagrange * len(sub_b)
        if RDO1 > RDO2: # Split mode
            result_lists[result_idx].append(sub_mae)
            result_lists[result_idx].append(1)
            result_lists[result_idx].append([[x, y, sub_best_mv[0][0], sub_best_mv[0][1], sub_best_ref_idx[0]],
                                             [x + I // 2, y, sub_best_mv[1][0], sub_best_mv[1][1], sub_best_ref_idx[1]],
                                             [x, y + I // 2, sub_best_mv[2][0], sub_best_mv[2][1], sub_best_ref_idx[2]],
                                             [x + I // 2, y + I // 2, sub_best_mv[3][0], sub_best_mv[3][1], sub_best_ref_idx[3]]])
            result_lists[result_idx].append([[x, y, sub_residual_block[0]],
                                             [x + I // 2, y, sub_residual_block[1]],
                                             [x, y + I // 2, sub_residual_block[2]],
                                             [x + I // 2, y + I // 2, sub_residual_block[3]]])
            result_lists[result_idx].append(sub_b)
            reconstructed_frame[y:y+I, x:x+I] = sub_reconstructed_block
            return
    result_lists[result_idx].append(mae)
    result_lists[result_idx].append(0)
    result_lists[result_idx].append([x, y, best_mv[0], best_mv[1], best_ref_idx])
    result_lists[result_idx].append([x, y, residual_block])
    if VBSEnable:
        b = golomb_encoding([0]) + b
    result_lists[result_idx].append(b)
    reconstructed_frame[y:y+I, x:x+I] = reconstructed_block
        
#   inter_processing_one_row CONSUMES:
# original_frame: (matrix of width * height)
# reference_frames: list of (matrix of width * height)
# y
# I: block size (integer)
# R: move range (integer)
# QTC
# QTC_sub
# VBSEnable
# Lagrange
# FMEEnable
# FastME
# ParallelMode
# result_list: []
#    inter_processing_one_row EFFECTS:
# 1. the row_result_list will be used to save generated data.
# row_result_list contains:
#   row_mae (float)
#   row_mv_list (list of [x, y, dx, dy, refFrame])
#   row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...] (I*I matrix)])
#   row_binary (binary file)
#   VBS_count (integer)
# 2. the reconstructed frame, will be modified.
def inter_processing_one_row(original_frame, reference_frames, y, I, R, QTC, QTC_sub, VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frame, row_result_list):
    row_nums = original_frame.shape[0] // I
    col_nums = original_frame.shape[1] // I
    total_count = row_nums * col_nums
    result_lists =  [[] for _ in range(total_count)]
    # result_lists contains:
    # A list of 
    #     [
    #     mae: float
    #     VBS_mode: 0/1 (not split/split)
    #     mv_list: [x, y, dx, dy, dref](not split mode) / [[x1, y1, dx1, dy1, dref1], [x2, y2, dx2, dy2, dref2], [x3, y3, dx3, dy3, dref3], [x4, y4, dx4, dy4, dref4]](split mode)
    #     residual_block: [x, y, residual block](not split mode) / [[x1, y1, residual block1], [x2, y2, residual block2], [x3, y3, residual block3], [x4, y4, residual block4]](split mode)
    #     block_binary: the binary includes mv_vector. (ParallelMode == 1: mv differential disable; Else: mv differential enable) (include VBS mode and mv and residual block)
    #     ]
    if ParallelMode == 1:
        thread_list = []
        for i in range(col_nums):
            thread = threading.Thread(target=inter_block_processing,
                                      args=(original_frame, reference_frames, [0, 0, 0], i * I, y, I, R, QTC, QTC_sub, VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frame, result_lists))
            thread.start()
            thread_list.append(thread)
        join_threads(thread_list)

    else:
        previous_mv = [0, 0, 0]
        for i in range(col_nums):
            inter_block_processing(original_frame, reference_frames, previous_mv, i * I, y, I, R, QTC, QTC_sub, VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frame, result_lists)
            current_idx = y // I * col_nums + i
            if result_lists[current_idx][1] == 0:
                previous_mv = [result_lists[current_idx][2][2], result_lists[current_idx][2][3], result_lists[current_idx][2][4]]
            else:
                previous_mv = [result_lists[current_idx][2][3][2], result_lists[current_idx][2][3][3], result_lists[current_idx][2][3][4]]

    # Integrate the data into the row_result_list.
    row_mae = 0.0
    row_mv_list = []
    row_residual_list = []
    row_binary = Bits()
    VBS_count = 0
    for i in range(y // I * col_nums, (y // I + 1) * col_nums):
        row_mae += result_lists[i][0]
        if result_lists[i][1] == 0: # Not split mode
            row_mv_list.append(result_lists[i][2])
            row_residual_list.append(result_lists[i][3])
            row_binary += result_lists[i][4]
        else:   # Split mode
            row_mv_list.append(result_lists[i][2][0])
            row_mv_list.append(result_lists[i][2][1])
            row_mv_list.append(result_lists[i][2][2])
            row_mv_list.append(result_lists[i][2][3])
            row_residual_list.append(result_lists[i][3][0])
            row_residual_list.append(result_lists[i][3][1])
            row_residual_list.append(result_lists[i][3][2])
            row_residual_list.append(result_lists[i][3][3])
            row_binary += result_lists[i][4]
            VBS_count += 1
    row_result_list.append(row_mae / col_nums)
    row_result_list.append(row_mv_list)
    row_result_list.append(row_residual_list)
    row_result_list.append(row_binary)
    row_result_list.append(VBS_count)

#   Inter_processing CONSUMES:
# current_frame: (matrix of width * height)
# reference_frames: list of (matrix of width * height)
# I: block size (integer)
# R: move range (integer)
# QTC
# QTC_sub
# VBSEnable
# Lagrange
# FMEEnable
# FastME
# ParallelMode
#   Inter_processing PRODUCES:
# frame_mae: the mae of this frame
# mv_list: list of [x, y, dx, dy, dref]
# residual_list: [x, y, residual block(matrix of I * I)]
# binary: binary file for this frame. (if ParallelMode is 1, we do not use the differential for the mv)
# reconstructed_frame: (matrix of width * height)
# VBS_percentage: float
def inter_processing(current_frame, reference_frames, I, R, QTC, QTC_sub, VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode):
    frame_mae = 0.0
    mv_list = []
    residual_list = []
    reconstructed_frame = np.zeros((current_frame.shape[0], current_frame.shape[1]), dtype=np.uint8)
    binary = Bits()
    sub_count = 0
    VBS_percentage = 0.0
    row_nums = current_frame.shape[0] // I
    col_nums = current_frame.shape[1] // I
    total_count = row_nums * col_nums
    
    if ParallelMode == 0:
        for y in range(0, current_frame.shape[0], I):
            previous_mv = (0, 0, 0)  # Initialize with a third component for the reference frame index
            for x in range(0, current_frame.shape[1], I):
                if(VBSEnable):
                    mae1, best_mv1, best_ref_idx1, residual_block1, reconstructed_block1, b1 = inter_full_block_processing(current_frame, reference_frames, previous_mv, x, y, I, R, QTC, FMEEnable, FastME)
                    mae2, best_mv2, best_ref_idx2, residual_block2, reconstructed_block2, b2 = inter_sub_block_processing(current_frame, reference_frames, previous_mv, x, y, I//2, R, QTC_sub, FMEEnable, FastME)
                    
                    SAD1 = np.sum(np.abs(current_frame[y:y+I, x:x+I] - reconstructed_block1))
                    SAD2 = np.sum(np.abs(current_frame[y:y+I, x:x+I] - reconstructed_block2))
                    RDO1 = SAD1 + Lagrange * len(golomb_encoding([0]) + b1)
                    RDO2 = SAD2 + Lagrange * len(golomb_encoding([1]) + b2)
                    if(RDO1 > RDO2):
                        previous_mv = (best_mv2[3][0], best_mv2[3][1], best_ref_idx2[3])
                        binary += golomb_encoding([1]) #split
                        binary += b2
                        
                        frame_mae += mae2
                        reconstructed_frame[y:y+I, x:x+I] = reconstructed_block2
                        mv_list.append((x, y, best_mv2[0][0], best_mv2[0][1], best_ref_idx2[0]))
                        mv_list.append((x+I//2, y, best_mv2[1][0], best_mv2[1][1], best_ref_idx2[1]))
                        mv_list.append((x, y+I//2, best_mv2[2][0], best_mv2[2][1], best_ref_idx2[2]))
                        mv_list.append((x+I//2, y+I//2, best_mv2[3][0], best_mv2[3][1], best_ref_idx2[3]))
                        residual_list.append((x, y, residual_block2[0]))
                        residual_list.append((x+I//2, y, residual_block2[1]))
                        residual_list.append((x, y+I//2, residual_block2[2]))
                        residual_list.append((x+I//2, y+I//2, residual_block2[3]))
                        sub_count += 1
                        
                    else:
                        previous_mv = (best_mv1[0], best_mv1[1], best_ref_idx1)
                        binary += golomb_encoding([0]) #no split
                        binary += b1
                        
                        frame_mae += mae1
                        
                        # Note: inter_sub_block_processing overwites the reconstructed_frame, write it back
                        reconstructed_frame[y:y+I, x:x+I] = reconstructed_block1
                        
                        mv_list.append((x, y, best_mv1[0], best_mv1[1], best_ref_idx1))
                        residual_list.append((x, y, residual_block1))
                else:
                    mae, best_mv, best_ref_idx, residual_block, reconstructed_block, b = inter_full_block_processing(current_frame, reference_frames, previous_mv, x, y, I, R, QTC, FMEEnable, FastME)
                    previous_mv = (best_mv[0], best_mv[1], best_ref_idx)
                    binary += b
                    
                    reconstructed_frame[y:y+I, x:x+I] = reconstructed_block
                    
                    frame_mae += mae
                    mv_list.append((x, y, best_mv[0], best_mv[1], best_ref_idx))
                    residual_list.append((x, y, residual_block))
        frame_mae /= total_count
        if VBSEnable:
            VBS_percentage = sub_count/total_count
    else:
        thread_list = []
        rows_result_lists = [[] for _ in range(row_nums)]
        # row_result_lists contains:
        # A list of
        #   [
        #   0: row_mae (float)
        #   1: row_mv_list (list of [x, y, dx, dy, refFrame])
        #   2: row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...]])
        #   3: row_binary (binary file)
        #   4: VBS_count (integer)
        #   ]
        for row in range(row_nums):
            thread = threading.Thread(target=inter_processing_one_row,
                                      args=(current_frame, reference_frames, row * I, I, R, QTC, QTC_sub, VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frame, rows_result_lists[row]))
            thread.start()
            thread_list.append(thread)
        join_threads(thread_list)

        for row_result_list in rows_result_lists:
            frame_mae += row_result_list[0]
            mv_list +=  row_result_list[1]
            residual_list += row_result_list[2]
            binary += row_result_list[3]
            sub_count += row_result_list[4]
        frame_mae /= row_nums
        if VBSEnable:
            VBS_percentage = sub_count / total_count
    
    return frame_mae, mv_list, residual_list, binary, reconstructed_frame, VBS_percentage

#   Inter_rows_processing CONSUMES:
# original_frame
# reference_frames
# I
# R
# QTC
# QTC_sub
# VBSEnable
# Lagrange
# FMEEnable
# FastME
# ParallelMode
# result_list
#   Inter_rows_processing EFFECTS:
# the result_list will be used to save the results:
# [
#   [row0_mae, row1_mae, row2_mae...], (float list)
#   [row0_mv_list, row1_mv_list, row2_mv_list...], (list of [x, y, dx, dy, dref])
#   [row0_residual_list, row1_residual_list, row2_residual_list...], (list of [x, y [[a1,b1,c1,d1...], [a2,b2,c2,d2...]...] (I * I or (I//2 * I//2))])
#   [row0_binary, row1_binary, row2_binary...], (binary list) (only use differential in each row.)
#   [reconstructed_row0, reconstructed_row1, reconstructed_row2], (list of [[a1,b1,c1,d1...z1], [a2,b2,c2,d2...]...] (I * WIDTH))
#   [row0_VBS_count, row1_VBS_count, row2_VBS_count...] (integer list)
# ]
def inter_rows_processing(original_frame, reference_frames, I, R, QTC, QTC_sub, VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, result_list):
    frame_mae = []
    mv_list = []
    residual_list = []
    binary_list = []
    reconstructed_list = []
    reconstructed_frame = np.zeros((original_frame.shape[0], original_frame.shape[1]), dtype=np.uint8)
    VBS_count = []
    row_nums = original_frame.shape[0] // I

    if ParallelMode == 0:
        for y in range(0, original_frame.shape[0], I):
            row_result_list = []
            # row_result_list contains:
            #   row_mae (float)
            #   row_mv_list (list of [x, y, dx, dy, refFrame])
            #   row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...] (I*I matrix)])
            #   row_binary (binary file)
            #   VBS_count (integer)
            inter_processing_one_row(original_frame, reference_frames, y, I, R, QTC, QTC_sub, VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frame, row_result_list)
            frame_mae.append(row_result_list[0])
            mv_list.append(row_result_list[1])
            residual_list.append(row_result_list[2])
            binary_list.append(row_result_list[3])
            reconstructed_list.append(reconstructed_frame[y:y+I,:])
            VBS_count.append(row_result_list[4])

    else:
        thread_list = []
        rows_result_lists = [[] for _ in range(row_nums)]
        # row_result_lists contains:
        # A list of
        #   [
        #   0: row_mae (float)
        #   1: row_mv_list (list of [x, y, dx, dy, refFrame])
        #   2: row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...]])
        #   3: row_binary (binary file)
        #   4: VBS_count (integer)
        #   ]
        for row in range(row_nums):
            thread = threading.Thread(target=inter_processing_one_row,
                                      args=(original_frame, reference_frames, row * I, I, R, QTC, QTC_sub, VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frame, rows_result_lists[row]))
            thread.start()
            thread_list.append(thread)
        join_threads(thread_list)

        for row in range(row_nums):
            row_result_list = rows_result_lists[row]
            y = row * I
            frame_mae.append(row_result_list[0])
            mv_list.append(row_result_list[1])
            residual_list.append(row_result_list[2])
            binary_list.append(row_result_list[3])
            reconstructed_list.append(reconstructed_frame[y:y+I,:])
            VBS_count.append(row_result_list[4])
    
    result_list.append(frame_mae)
    result_list.append(mv_list)
    result_list.append(residual_list)
    result_list.append(binary_list)
    result_list.append(reconstructed_list)
    result_list.append(VBS_count)

#   Encode_frame function CONSUMES:
# file_dir: directory for the original y-frames.
# frame_num: idx of the frame encoding now.
# WIDTH
# HEIGHT
# I: block size
# R: search range
# QP: quantization parameter
# QTCs: a list of QTCs with block size I.
# QTC_subs: a list of QTCs with block size I // 2.
# I_PERIOD: the period of i frames. (0, i-1, 2i-1, 3i-1) are all intra frames.
# OUTPUT_DIR
# reference_frames: reconstructed frames.
# VBSEnable
# FMEEnable
# FastME
# Lagrange: intra lagrange
# Lagrange2: inter lagrange
# RCflag
# frame_BR: bit rate for this frame.
# ParallelMode
#   Encode_frame function PRODUCES:
# frame_mae: the Mean of Absolute Error
# len(frame_binary): this binary file's length
# encoding_time
# reference_frames: reconstructed frames.
# VBS_percentage
#   Encode_frame function EFFECTS:
# saves files: frame_x_mode/motion_vector.txt; reconstructed_frame; residual list; binary frame.
def encode_frame(file_dir, frame_num, table_frame_nums, width, height, I, R, QP, QTCs, QTC_subs, I_period, output_dir, reference_frames, VBSEnable, FMEEnable, FastME, Lagrange, Lagrange2, RCflag, frame_BR, ParallelMode):
    start_time = time.time()
    # Load original frame.
    original_frame = read_and_pad(f"{file_dir}/y_frame_{frame_num}.y", I, width, height)
    frame_binary = Bits()
    frame_mae = 0.0
    mode_mv_list = []
    residual_list = []
    reconstructed_frame = np.full((height, width), 128, dtype=np.uint8)
    VBS_percentage = 0.0
    current_QPs = []
    is_intra_frame = (I_period == 0 or frame_num % I_period == 0) and (not (ParallelMode == 1))

    if RCflag == 0: # Blind mode. (Constant QP)
        if is_intra_frame:  # Intra frame
            frame_binary += '0b1'
            QP_sub = QP if QP == 0 else QP - 1
            frame_mae, mode_list, residual_list, binary, reconstructed_frame, VBS_percentage = intra_processing(original_frame, I, QTCs[QP], QTC_subs[QP_sub], VBSEnable, Lagrange, ParallelMode)
            frame_binary += binary
            # Reset reference frames to only contain the newly reconstructed intra frame
            if(FMEEnable):
                FME_reconstructed_frame = frame_interpolation(reconstructed_frame, I)
                reference_frames = [FME_reconstructed_frame for _ in range(len(reference_frames))]
            else:
                reference_frames = [reconstructed_frame for _ in range(len(reference_frames))]
            # Save mode and motion vectors to txt
            mode_file_path = os.path.join(output_dir, 'mode_and_motion_vector', f'frame_{frame_num}_mode.txt')
            os.makedirs(os.path.dirname(mode_file_path), exist_ok=True)
            with open(mode_file_path, 'w') as f:
                for val in mode_list:
                    f.write(f"{val[0]} {val[1]} {val[2]}\n")
        else:   # Inter frame.
            frame_binary += '0b0'
            QP_sub = QP if QP == 0 else QP - 1
            frame_mae, mv_list, residual_list, binary, reconstructed_frame, VBS_percentage = inter_processing(original_frame, reference_frames, I, R, QTCs[QP], QTC_subs[QP_sub], VBSEnable, Lagrange2, FMEEnable, FastME, ParallelMode)
            frame_binary += binary
            # Update reference frames list, remove the oldest reference frame and add the new one
            reference_frames.pop(0)
            if(FMEEnable):
                reference_frames.append(frame_interpolation(reconstructed_frame, I))
            else:
                reference_frames.append(reconstructed_frame)
            # Save mode and motion vectors vectors to txt
            mv_file_path = os.path.join(output_dir, 'mode_and_motion_vector', f'frame_{frame_num}_motion_vector.txt')
            os.makedirs(os.path.dirname(mv_file_path), exist_ok=True)
            with open(mv_file_path, 'w') as f:
                for val in mv_list:
                    f.write(f"{val[0]} {val[1]} {val[2]} {val[3]} {val[4]}\n")  # Include reference frame index


    if RCflag == 1:
        previous_QP = 0
        BR_table = []
        # Add frame type bit to binary and load the table.
        if is_intra_frame:  # Intra frame
            frame_binary += '0b1'
            with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Intra.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                next(reader)
                next(reader)
                BR_table = np.array([int(val) for val in next(reader)])
                
        if not is_intra_frame:  # Inter frame
            frame_binary += '0b0'
            with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Inter.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                next(reader)
                next(reader)
                BR_table = np.array([int(val) for val in next(reader)])
        
        # For each row, we arrange a BR for it and use the table to find related QP to encode it.
        for row in range(height // I):
            row_BR = (frame_BR - len(frame_binary)) // (height // I - row)
            # Find the QP and the QP_sub for this row.
            row_QP = 0
            if BR_table[0] > row_BR:
                QP_nums = len(BR_table)
                for i in range(QP_nums):
                    if BR_table[i] > row_BR:
                        row_QP = i + 1
                    if row_QP >= QP_nums:
                        row_QP = QP_nums - 1
            row_QP_sub = row_QP if row_QP == 0 else row_QP - 1
            # encode this row with the row_QP.
            row_result_list = []
            # row_result_list contains:
            # 0: row_mae (float)
            # 1: row_mode_list (list of [x, y, (0(horizontal)/1(vertical))]) / (list of [x, y, dx, dy, refFrame])
            # 2: row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...]])
            # 3: row_binary (binary file)
            # 4: VBS_count (integer)

            if is_intra_frame:
                intra_processing_one_row(original_frame, reconstructed_frame, row * I, I, QTCs[row_QP], QTC_subs[row_QP_sub], VBSEnable, Lagrange, ParallelMode, row_result_list)
            if not is_intra_frame:
                inter_processing_one_row(original_frame, reference_frames, row * I, I, R, QTCs[row_QP], QTC_subs[row_QP_sub], VBSEnable, Lagrange2, FMEEnable, FastME, ParallelMode, reconstructed_frame, row_result_list)
            frame_mae += row_result_list[0]
            mode_mv_list += row_result_list[1]
            residual_list += row_result_list[2]
            diff_QP = row_QP - previous_QP
            previous_QP = row_QP
            frame_binary += golomb_encoding([diff_QP]) + row_result_list[3]
            VBS_percentage += row_result_list[4]
            
        if is_intra_frame:
            # Reset reference frames to only contain the newly reconstructed intra frame
            if(FMEEnable):
                FME_reconstructed_frame = frame_interpolation(reconstructed_frame, I)
                reference_frames = [FME_reconstructed_frame for _ in range(len(reference_frames))]
            else:
                reference_frames = [reconstructed_frame for _ in range(len(reference_frames))]
            # Save mode and motion vectors to txt
            mode_file_path = os.path.join(output_dir, 'mode_and_motion_vector', f'frame_{frame_num}_mode.txt')
            os.makedirs(os.path.dirname(mode_file_path), exist_ok=True)
            with open(mode_file_path, 'w') as f:
                for val in mode_mv_list:
                    f.write(f"{val[0]} {val[1]} {val[2]}\n")
        if not is_intra_frame:
            # Update reference frames list, remove the oldest reference frame and add the new one
            reference_frames.pop(0)
            if(FMEEnable):
                reference_frames.append(frame_interpolation(reconstructed_frame, I))
            else:
                reference_frames.append(reconstructed_frame)
            # Save mode and motion vectors vectors to txt
            mv_file_path = os.path.join(output_dir, 'mode_and_motion_vector', f'frame_{frame_num}_motion_vector.txt')
            os.makedirs(os.path.dirname(mv_file_path), exist_ok=True)
            with open(mv_file_path, 'w') as f:
                for val in mode_mv_list:
                    f.write(f"{val[0]} {val[1]} {val[2]} {val[3]} {val[4]}\n")  # Include reference frame index
        frame_mae /= height // I
        VBS_percentage /= (width // I) * (height // I)


    if RCflag == 2 or RCflag == 3:
        # First pass
        # result_list contains:
        # 0: [row0_mae, row1_mae, row2_mae...], (float list)
        # 1: [row0_mode_list, row1_mode_list, row2_mode_list...], (list of [x, y, (0(horizontal)/1(vertical))/(dx, dy, dref_frame)])
        # 2: [row0_residual_list, row1_residual_list, row2_residual_list...], (list of [x, y [[a1,b1,c1,d1...], [a2,b2,c2,d2...]...] (I * I or (I//2 * I//2))])
        # 3: [row0_binary, row1_binary, row2_binary...], (binary list)
        # 4: [reconstructed_row0, reconstructed_row1, reconstructed_row2], (list of [[a1,b1,c1,d1...z1], [a2,b2,c2,d2...]...] (I * WIDTH))
        # 5: [row0_VBS_count, row1_VBS_count, row2_VBS_count...] (integer list)
        result_list = []
        QP_sub = QP if QP == 0 else QP - 1
        scene_change = False
        if is_intra_frame:
            intra_rows_processing(original_frame, I, QTCs[QP], QTC_subs[QP_sub], VBSEnable, Lagrange, ParallelMode, result_list)
        if not is_intra_frame:
            inter_rows_processing(original_frame, reference_frames, I, R, QTCs[QP], QTC_subs[QP_sub], VBSEnable, Lagrange2, FMEEnable, FastME, ParallelMode, result_list)
            if sum(len(binary_str) for binary_str in result_list[3]) >= scene_change_threshold(width, height, QP) and (not ParallelMode == 1):    # ParallelMode 1 disabled the intra frame which is scene_change
                scene_change = True
        
        # Second pass
        previous_QP = 0
        BR_table = []
        if is_intra_frame or scene_change:
            frame_binary += '0b1'
            with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Intra.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                next(reader)
                next(reader)
                BR_table = np.array([int(val) for val in next(reader)])
        if not (is_intra_frame or scene_change):
            frame_binary += '0b0'
            with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Intra.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                next(reader)
                next(reader)
                BR_table = np.array([int(val) for val in next(reader)])
        # Get the proportion table.
        proportion_table = []
        rows_BR = []
        for row_binary in result_list[3]:
            rows_BR.append(len(row_binary))
        total_BR = np.sum(rows_BR)
        for row_BR in rows_BR:
            proportion_table.append(row_BR / total_BR)
        # Get the new BR table for this frame.
        average_row_BR = total_BR / (height // I)
        scaling_factor = average_row_BR / BR_table[QP]
        for i in range(len(BR_table)):
            BR_table[i] *= scaling_factor
        # For each row, we arrange a BR for it and use the table to find related QP to encode it.
        for row in range(height // I):
            row_BR = proportion_table[row] * frame_BR
            # Find the QP and the QP_sub for this row.
            row_QP = 0
            if BR_table[0] > row_BR:
                QP_nums = len(BR_table)
                for i in range(QP_nums):
                    if BR_table[i] > row_BR:
                        row_QP = i + 1
                    if row_QP >= QP_nums:
                        row_QP = QP_nums - 1
            current_QPs.append(row_QP)
            # encode this row with the row_QP.
            row_result_list = []
            # row_result_list contains:
            # 0: row_mae (float)
            # 1: row_mode_list (list of [x, y, (0(horizontal)/1(vertical))]) / (list of [x, y, dx, dy, refFrame])
            # 2: row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...]])
            # 3: row_binary (binary file)
            # 4: VBS_count (integer)
            if (not is_intra_frame) and row_QP == QP and (not scene_change) and RCflag == 3: # Can benefit from first pass
                row_result_list.append(result_list[0][row])
                row_result_list.append(result_list[1][row])
                row_result_list.append(result_list[2][row])
                row_result_list.append(result_list[3][row])
                row_result_list.append(result_list[5][row])
                reconstructed_frame[row*I:(row+1)*I, :] = result_list[4][row]
            if not ((not is_intra_frame) and row_QP == QP and (not scene_change) and RCflag == 3):
                row_QP_sub = row_QP if row_QP == 0 else row_QP - 1
                if is_intra_frame or scene_change:
                    intra_processing_one_row(original_frame, reconstructed_frame, row * I, I, QTCs[row_QP], QTC_subs[row_QP_sub], VBSEnable, Lagrange, ParallelMode, row_result_list)
                if not (is_intra_frame or scene_change):
                    inter_processing_one_row(original_frame, reference_frames, row * I, I, R, QTCs[row_QP], QTC_subs[row_QP_sub], VBSEnable, Lagrange2, FMEEnable, FastME, ParallelMode, reconstructed_frame, row_result_list)
            frame_mae += row_result_list[0]
            mode_mv_list += row_result_list[1]
            residual_list += row_result_list[2]
            diff_QP = row_QP - previous_QP
            previous_QP = row_QP
            frame_binary += golomb_encoding([diff_QP]) + row_result_list[3]
            VBS_percentage += row_result_list[4]
        if is_intra_frame or scene_change:
            # Reset reference frames to only contain the newly reconstructed intra frame
            if(FMEEnable):
                FME_reconstructed_frame = frame_interpolation(reconstructed_frame, I)
                reference_frames = [FME_reconstructed_frame for _ in range(len(reference_frames))]
            else:
                reference_frames = [reconstructed_frame for _ in range(len(reference_frames))]
            # Save mode and motion vectors to txt
            mode_file_path = os.path.join(output_dir, 'mode_and_motion_vector', f'frame_{frame_num}_mode.txt')
            os.makedirs(os.path.dirname(mode_file_path), exist_ok=True)
            with open(mode_file_path, 'w') as f:
                for val in mode_mv_list:
                    f.write(f"{val[0]} {val[1]} {val[2]}\n")
        if not (is_intra_frame or scene_change):
            # Update reference frames list, remove the oldest reference frame and add the new one
            reference_frames.pop(0)
            if(FMEEnable):
                reference_frames.append(frame_interpolation(reconstructed_frame, I))
            else:
                reference_frames.append(reconstructed_frame)
            # Save mode and motion vectors vectors to txt
            mv_file_path = os.path.join(output_dir, 'mode_and_motion_vector', f'frame_{frame_num}_motion_vector.txt')
            os.makedirs(os.path.dirname(mv_file_path), exist_ok=True)
            with open(mv_file_path, 'w') as f:
                for val in mode_mv_list:
                    f.write(f"{val[0]} {val[1]} {val[2]} {val[3]} {val[4]}\n")  # Include reference frame index
        frame_mae /= height // I
        VBS_percentage /= (width // I) * (height // I)

    # Save the reconstructed frame
    reconstructed_file_path = os.path.join(output_dir, 'y_frames_reconstructed', f'y_reconstructed_frame_{frame_num}.y')
    os.makedirs(os.path.dirname(reconstructed_file_path), exist_ok=True)
    with open(reconstructed_file_path, 'wb') as f:
        f.write(reconstructed_frame[:height, :width].tobytes())

    # Save the residual values
    residuals_file_path = os.path.join(output_dir, 'residuals', f'residuals_frame_{frame_num}.txt')
    os.makedirs(os.path.dirname(residuals_file_path), exist_ok=True)
    with open(residuals_file_path, 'w') as f:
        for res in residual_list:
            residual_str = json.dumps(res[2].tolist())
            f.write(f"{res[0]} {res[1]} {residual_str}\n")

    # Save the encoded binary
    binary_file_path = os.path.join(output_dir, 'encoded_binary', f'binary_frame_{frame_num}')
    os.makedirs(os.path.dirname(binary_file_path), exist_ok=True)
    with open(binary_file_path, 'wb') as f:
        frame_binary.tofile(f)

    if RCflag == 2 or RCflag == 3:
        QP = int(np.mean(current_QPs))

    encoding_time = time.time() - start_time

    return frame_mae, len(frame_binary), encoding_time, reference_frames, VBS_percentage, QP

#   first_frame function CONSUMES:
# original_first_frame: width * height matrix
# table_frame_nums
# width
# height
# I: block size (int)
# R: search range(int)
# next_QP: [QP]
# QTCs
# QTC_subs
# I_period
# output_dir
# reference_frames: the reference frames might be used, and 2 more spaces for this and next reconstructed frames.
# VBSEnable
# FMEEnable
# FastME
# Lagrange
# Lagrange2
# RCflag
# frame_BR
# ParallelMode
# event_list: A list of threading.Event() with length of number of rows. (Set the corresponding event after encoding the row.)
# maes
# binary_lengths
# VBS_percentages
#   first_frame funtion EFFECTS:
# 1. next_QP would be changed after the first pass is done. (average of all rows' QPs if RCflag == 2 / 3)
# 2. reference_frames would be modified: The penultimate frame would be changed as reconstructed frame.
# 3. event_list would be changed: The event_list would be set after reconstructing the corresponding row.
# 4. updates maes, binary_lengths, VBS_percentages
# 5. saves files: frame_x_mode/motion_vector.txt; reconstructed_frame; residual list; binary frame.
def first_frame(original_first_frame, frame_num, table_frame_nums, width, height, I, R, next_QP, QTCs, QTC_subs, I_period, output_dir, reference_frames, VBSEnable, FMEEnable, FastME, Lagrange, Lagrange2, RCflag, frame_BR, ParallelMode, event_list, maes, binary_lengths, VBS_percentages):
    QP = next_QP[0]
    row_nums = height // I
    col_nums = width // I
    total_count = row_nums * col_nums
    frame_mae = 0.0
    VBS_count = 0
    mode_mv_list = []
    reconstructed_frame = np.full((height, width), 128, dtype=np.uint8)
    residual_list = []
    frame_binary = Bits()
    is_intra_frame = I_period == 0 or frame_num % I_period == 0
    scene_change = False

    if RCflag == 0:
        QP_sub = QP if QP == 0 else QP - 1
        
        if is_intra_frame:
            frame_binary += '0b1'
            result_lists = [[] for _ in range(total_count)]
            # result_lists contains:
            # A list of 
            # [
            # 0: mae: float
            # 1: VBS mode: 0(not split)/1(split)
            # 2: hori_verti mode: [x, y, (0(horizontal)/1(vertical))] / [[x1, y1, 0/1], [x2, y2, 0/1], [x3, y3, 0/1], [x4, y4, 0/1]](split mode)
            # 3: residual block: [x, y, I * I matrix] / [[x1, y1, I/2 * I/2 matrix], [x2, y2, I/2 * I/2 matrix], [x3, y3, I/2 * I/2 matrix], [x4, y4, I/2 * I/2 matrix]]
            # 4: block binary
            # ]
            total_layer = row_nums + col_nums - 1
            for i in range(total_layer):
                thread_list = []
                for x in range(min(i + 1, col_nums)):
                    y = i - x
                    if y >= 0 and y < row_nums:    # Valid y column
                        thread = threading.Thread(target=intra_block_processing,
                                                  args=(original_first_frame, QTCs[QP], QTC_subs[QP_sub], reconstructed_frame, x * I, y * I, I, VBSEnable, Lagrange, ParallelMode, result_lists))
                        thread.start()
                        thread_list.append(thread)
                if x == col_nums - 1:
                    interpolated_frame = reconstructed_frame
                    if FMEEnable:
                        interpolated_frame = frame_interpolation(reconstructed_frame, I)
                    for j in range(len(reference_frames)):
                        reference_frames[j] = interpolated_frame
                    event_list[y].set()
                join_threads(thread_list)
            # Load data: maes, binary_lengths, VBS_percentages, mode_mv_list, residual_list, frame_binary
            for result_list in result_lists:
                frame_mae += result_list[0]
                frame_binary += result_list[4]
                if result_list[1] == 0:
                    mode_mv_list.append(result_list[2])
                    residual_list.append(result_list[3])
                else:
                    mode_mv_list.append(result_list[2][0])
                    mode_mv_list.append(result_list[2][1])
                    mode_mv_list.append(result_list[2][2])
                    mode_mv_list.append(result_list[2][3])
                    residual_list.append(result_list[3][0])
                    residual_list.append(result_list[3][1])
                    residual_list.append(result_list[3][2])
                    residual_list.append(result_list[3][3])
                    VBS_count += 1
            maes.append(frame_mae / total_count)
            binary_lengths.append(len(frame_binary))
            VBS_percentages.append(VBS_count / total_count)
        
        else:
            frame_binary += '0b0'
            for row in range(row_nums):
                row_result_list = []
                # row_result_list contains:
                #   0: row_mae (float)
                #   1: row_mv_list (list of [x, y, dx, dy, refFrame])
                #   2: row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...] (I*I matrix)])
                #   3: row_binary (binary file)
                #   4: VBS_count (integer)
                slice_reference_frames = reference_frames[:-2]
                inter_processing_one_row(original_first_frame, slice_reference_frames, row * I, I, R, QTCs[QP], QTC_subs[QP_sub], VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frame, row_result_list)
                interpolated_frame = reconstructed_frame
                if FMEEnable:
                    interpolated_frame = frame_interpolation(reconstructed_frame, I)
                reference_frames[-2] = interpolated_frame
                event_list[row].set()
                # Load data: maes, binary_lengths, VBS_percentages, mode_mv_list, residual_list, frame_binary
                frame_mae += row_result_list[0]
                frame_binary += row_result_list[3]
                VBS_count += row_result_list[4]
                mode_mv_list += row_result_list[1]
                residual_list += row_result_list[2]
            maes.append(frame_mae / row_nums)
            binary_lengths.append(len(frame_binary))
            VBS_percentages.append(VBS_count / total_count)

    if RCflag == 1:
        previous_QP = 0
        BR_table = []
        # Add frame type bit to binary and load the table.
        if is_intra_frame:  # Intra frame
            frame_binary += '0b1'
            with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Intra.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                next(reader)
                next(reader)
                BR_table = np.array([int(val) for val in next(reader)])
                
        if not is_intra_frame:  # Inter frame
            frame_binary += '0b0'
            with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Inter.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                next(reader)
                next(reader)
                BR_table = np.array([int(val) for val in next(reader)])

        # For each row, we arrange a BR for it and use the table to find related QP to encode it.
        for row in range(height // I):
            row_BR = (frame_BR - len(frame_binary)) // (height // I - row)
            # Find the QP and the QP_sub for this row.
            row_QP = 0
            if BR_table[0] > row_BR:
                QP_nums = len(BR_table)
                for i in range(QP_nums):
                    if BR_table[i] > row_BR:
                        row_QP = i + 1
                    if row_QP >= QP_nums:
                        row_QP = QP_nums - 1
            row_QP_sub = row_QP if row_QP == 0 else row_QP - 1
            row_result_list = []
            # row_result_list contains:
            # [
            #   0: row_mae (float)
            #   1: row_mode_list (list of [x, y, (0(horizontal)/1(vertical))]))
            #   2: row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...]])
            #   3: row_binary (binary file) (VBS mode and hori_verti mode are both differential, started from the first element of this row.)
            #   4: VBS_count (integer)
            # ]
            if is_intra_frame:
                intra_processing_one_row(original_first_frame, reconstructed_frame, row * I, I, QTCs[row_QP], QTC_subs[row_QP_sub], VBSEnable, Lagrange, ParallelMode, row_result_list)
                interpolated_frame = reconstructed_frame
                if FMEEnable:
                    interpolated_frame = frame_interpolation(reconstructed_frame, I)
                for j in range(len(reference_frames)):
                    reference_frames[j] = interpolated_frame
            if not is_intra_frame:
                inter_processing_one_row(original_first_frame, reference_frames[:-2], row * I, I, R, QTCs[row_QP], QTC_subs[row_QP_sub], VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frame, row_result_list)
                interpolated_frame = reconstructed_frame
                if FMEEnable:
                    interpolated_frame = frame_interpolation(reconstructed_frame, I)
                reference_frames[-2] = interpolated_frame
            event_list[row].set()

            # Load data: maes, binary_lengths, VBS_percentages, mode_mv_list, residual_list, frame_binary
            frame_mae += row_result_list[0]
            diff_QP = row_QP - previous_QP
            previous_QP = row_QP
            frame_binary += golomb_encoding([diff_QP]) + row_result_list[3]
            VBS_count += row_result_list[4]
            mode_mv_list += row_result_list[1]
            residual_list += row_result_list[2]
        maes.append(frame_mae / row_nums)
        binary_lengths.append(len(frame_binary))
        VBS_percentages.append(VBS_count / total_count)

    if RCflag == 2 or RCflag == 3:
        # First pass
        result_list = []
        # result_list contains:
        # 0: [row0_mae, row1_mae, row2_mae...], (float list)
        # 1: [row0_mode_list, row1_mode_list, row2_mode_list...], (list of [x, y, (0(horizontal)/1(vertical))/(dx, dy, dref_frame)])
        # 2: [row0_residual_list, row1_residual_list, row2_residual_list...], (list of [x, y [[a1,b1,c1,d1...], [a2,b2,c2,d2...]...] (I * I or (I//2 * I//2))])
        # 3: [row0_binary, row1_binary, row2_binary...], (binary list)
        # 4: [reconstructed_row0, reconstructed_row1, reconstructed_row2], (list of [[a1,b1,c1,d1...z1], [a2,b2,c2,d2...]...] (I * WIDTH))
        # 5: [row0_VBS_count, row1_VBS_count, row2_VBS_count...] (integer list)
        QP_sub = QP if QP == 0 else QP - 1
        scene_change = False
        if is_intra_frame:
            intra_rows_processing(original_first_frame, I, QTCs[QP], QTC_subs[QP_sub], VBSEnable, Lagrange, ParallelMode, result_list)
        if not is_intra_frame:
            inter_rows_processing(original_first_frame, reference_frames[:-2], I, R, QTCs[QP], QTC_subs[QP_sub], VBSEnable, Lagrange2, FMEEnable, FastME, ParallelMode, result_list)
            if sum(len(binary_str) for binary_str in result_list[3]) >= scene_change_threshold(width, height, QP) and (not ParallelMode == 1):    # ParallelMode 1 disabled the intra frame which is scene_change
                scene_change = True

        # Second pass
        previous_QP = 0
        BR_table = []
        if is_intra_frame or scene_change:
            frame_binary += '0b1'
            with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Intra.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                next(reader)
                next(reader)
                BR_table = np.array([int(val) for val in next(reader)])
        if not (is_intra_frame or scene_change):
            frame_binary += '0b0'
            with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Intra.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                next(reader)
                next(reader)
                BR_table = np.array([int(val) for val in next(reader)])
        # Get the proportion table.
        proportion_table = []
        rows_BR = []
        for row_binary in result_list[3]:
            rows_BR.append(len(row_binary))
        total_BR = np.sum(rows_BR)
        for row_BR in rows_BR:
            proportion_table.append(row_BR / total_BR)
        # Get the new BR table for this frame.
        average_row_BR = total_BR / (height // I)
        scaling_factor = average_row_BR / BR_table[QP]
        for i in range(len(BR_table)):
            BR_table[i] *= scaling_factor
        # For each row, we arrange a BR for it and use the table to find related QP to encode it.
        row_QPs = []
        row_QP_subs = []
        for row in range(height // I):
            row_BR = proportion_table[row] * frame_BR
            # Find the QP and the QP_sub for this row.
            row_QP = 0
            if BR_table[0] > row_BR:
                QP_nums = len(BR_table)
                for i in range(QP_nums):
                    if BR_table[i] > row_BR:
                        row_QP = i + 1
                    if row_QP >= QP_nums:
                        row_QP = QP_nums - 1
            row_QPs.append(row_QP)
            row_QP_sub = row_QP if row_QP == 0 else row_QP - 1
            row_QP_subs.append(row_QP_sub)

        # update next QP value
        next_QP[0] = int(np.mean(row_QPs))

        for row in range(height // I):
            row_QP = row_QPs[row]
            row_QP_sub = row_QP_subs[row]
            # encode this row with the row_QP.
            row_result_list = []
            # row_result_list contains:
            # 0: row_mae (float)
            # 1: row_mode_list (list of [x, y, (0(horizontal)/1(vertical))]) / (list of [x, y, dx, dy, refFrame])
            # 2: row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...]])
            # 3: row_binary (binary file)
            # 4: VBS_count (integer)
            if (not is_intra_frame) and row_QP == QP and (not scene_change) and RCflag == 3: # Can benefit from first pass
                row_result_list.append(result_list[0][row])
                row_result_list.append(result_list[1][row])
                row_result_list.append(result_list[2][row])
                row_result_list.append(result_list[3][row])
                row_result_list.append(result_list[5][row])
                reconstructed_frame[row*I:(row+1)*I, :] = result_list[4][row]
            if not ((not is_intra_frame) and row_QP == QP and (not scene_change) and RCflag == 3):
                if is_intra_frame or scene_change:
                    intra_processing_one_row(original_first_frame, reconstructed_frame, row * I, I, QTCs[row_QP], QTC_subs[row_QP_sub], VBSEnable, Lagrange, ParallelMode, row_result_list)
                if not (is_intra_frame or scene_change):
                    inter_processing_one_row(original_first_frame, reference_frames[:-2], row * I, I, R, QTCs[row_QP], QTC_subs[row_QP_sub], VBSEnable, Lagrange2, FMEEnable, FastME, ParallelMode, reconstructed_frame, row_result_list)
            if is_intra_frame or scene_change:
                interpolated_frame = reconstructed_frame
                if FMEEnable:
                    interpolated_frame = frame_interpolation(reconstructed_frame, I)
                for j in range(len(reference_frames)):
                    reference_frames[j] = interpolated_frame
            if not (is_intra_frame or scene_change):
                interpolated_frame = reconstructed_frame
                if FMEEnable:
                    interpolated_frame = frame_interpolation(reconstructed_frame, I)
                reference_frames[-2] = interpolated_frame
            event_list[row].set()

            # Load data: maes, binary_lengths, VBS_percentages, mode_mv_list, residual_list, frame_binary
            frame_mae += row_result_list[0]
            diff_QP = row_QP - previous_QP
            previous_QP = row_QP
            frame_binary += golomb_encoding([diff_QP]) + row_result_list[3]
            VBS_count += row_result_list[4]
            mode_mv_list += row_result_list[1]
            residual_list += row_result_list[2]
        maes.append(frame_mae / row_nums)
        binary_lengths.append(len(frame_binary))
        VBS_percentages.append(VBS_count / total_count)

    if len(mode_mv_list) > 0 and len(mode_mv_list[0]) == 3:
        # Save mode to txt
            mode_file_path = os.path.join(output_dir, 'mode_and_motion_vector', f'frame_{frame_num}_mode.txt')
            os.makedirs(os.path.dirname(mode_file_path), exist_ok=True)
            with open(mode_file_path, 'w') as f:
                for val in mode_mv_list:
                    f.write(f"{val[0]} {val[1]} {val[2]}\n")
    if len(mode_mv_list) > 0 and len(mode_mv_list[0]) == 5:
        # Save mode and motion vectors vectors to txt
            mv_file_path = os.path.join(output_dir, 'mode_and_motion_vector', f'frame_{frame_num}_motion_vector.txt')
            os.makedirs(os.path.dirname(mv_file_path), exist_ok=True)
            with open(mv_file_path, 'w') as f:
                for val in mode_mv_list:
                    f.write(f"{val[0]} {val[1]} {val[2]} {val[3]} {val[4]}\n")  # Include reference frame index

    # Save the reconstructed frame
    reconstructed_file_path = os.path.join(output_dir, 'y_frames_reconstructed', f'y_reconstructed_frame_{frame_num}.y')
    os.makedirs(os.path.dirname(reconstructed_file_path), exist_ok=True)
    with open(reconstructed_file_path, 'wb') as f:
        f.write(reconstructed_frame[:height, :width].tobytes())
    
    # Save the residual values
    residuals_file_path = os.path.join(output_dir, 'residuals', f'residuals_frame_{frame_num}.txt')
    os.makedirs(os.path.dirname(residuals_file_path), exist_ok=True)
    with open(residuals_file_path, 'w') as f:
        for res in residual_list:
            residual_str = json.dumps(res[2].tolist())
            f.write(f"{res[0]} {res[1]} {residual_str}\n")
    
    # Save the encoded binary
    binary_file_path = os.path.join(output_dir, 'encoded_binary', f'binary_frame_{frame_num}')
    os.makedirs(os.path.dirname(binary_file_path), exist_ok=True)
    with open(binary_file_path, 'wb') as f:
        frame_binary.tofile(f)

#   second_frame function CONSUMES:
# original_first_frame: width * height matrix
# table_frame_nums
# width
# height
# I: block size (int)
# R: search range(int)
# next_QP: [QP]
# QTCs
# QTC_subs
# I_period
# output_dir
# reference_frames: the reference frames might be used, and 2 more spaces for this and previous reconstructed frames.
# VBSEnable
# FMEEnable
# FastME
# Lagrange
# Lagrange2
# RCflag
# frame_BR
# ParallelMode
# event_list: A list of threading.Event() with length of number of rows. (Check the event to do the encoding job.)
# maes
# binary_lengths
# VBS_percentages
#   first_frame funtion EFFECTS:
# 1. next_QP would be changed after the first pass is done. (average of all rows' QPs if RCflag == 2 / 3)
# 2. reference_frames would be modified: The last frame would be changed as reconstructed frame.
# 3. updates maes, binary_lengths, VBS_percentages
# 4. saves files: frame_x_mode/motion_vector.txt; reconstructed_frame; residual list; binary frame.
def second_frame(original_second_frame, frame_num, table_frame_nums, width, height, I, R, next_QP, QTCs, QTC_subs, I_period, output_dir, reference_frames, VBSEnable, FMEEnable, FastME, Lagrange, Lagrange2, RCflag, frame_BR, ParallelMode, event_list, maes, binary_lengths, VBS_percentages):
    QP = next_QP[0]
    row_nums = height // I
    col_nums = width // I
    total_count = row_nums * col_nums
    frame_mae = 0.0
    VBS_count = 0
    mode_mv_list = []
    reconstructed_frame = np.full((height, width), 128, dtype=np.uint8)
    residual_list = []
    frame_binary = Bits()
    is_intra_frame = I_period == 0 or frame_num % I_period == 0
    scene_change = False

    if RCflag == 0:
        QP_sub = QP if QP == 0 else QP - 1
        
        if is_intra_frame:
            frame_binary += '0b1'
            result_lists = [[] for _ in range(total_count)]
            # result_lists contains:
            # A list of 
            # [
            # 0: mae: float
            # 1: VBS mode: 0(not split)/1(split)
            # 2: hori_verti mode: [x, y, (0(horizontal)/1(vertical))] / [[x1, y1, 0/1], [x2, y2, 0/1], [x3, y3, 0/1], [x4, y4, 0/1]](split mode)
            # 3: residual block: [x, y, I * I matrix] / [[x1, y1, I/2 * I/2 matrix], [x2, y2, I/2 * I/2 matrix], [x3, y3, I/2 * I/2 matrix], [x4, y4, I/2 * I/2 matrix]]
            # 4: block binary
            # ]
            total_layer = row_nums + col_nums - 1
            for i in range(total_layer):
                thread_list = []
                for x in range(min(i + 1, col_nums)):
                    y = i - x
                    if y >= 0 and y < row_nums:    # Valid y column
                        thread = threading.Thread(target=intra_block_processing,
                                                  args=(original_second_frame, QTCs[QP], QTC_subs[QP_sub], reconstructed_frame, x * I, y * I, I, VBSEnable, Lagrange, ParallelMode, result_lists))
                        thread.start()
                        thread_list.append(thread)
                join_threads(thread_list)
            # Load data: maes, binary_lengths, VBS_percentages, mode_mv_list, residual_list, frame_binary
            for result_list in result_lists:
                frame_mae += result_list[0]
                frame_binary += result_list[4]
                if result_list[1] == 0:
                    mode_mv_list.append(result_list[2])
                    residual_list.append(result_list[3])
                else:
                    mode_mv_list.append(result_list[2][0])
                    mode_mv_list.append(result_list[2][1])
                    mode_mv_list.append(result_list[2][2])
                    mode_mv_list.append(result_list[2][3])
                    residual_list.append(result_list[3][0])
                    residual_list.append(result_list[3][1])
                    residual_list.append(result_list[3][2])
                    residual_list.append(result_list[3][3])
                    VBS_count += 1
            maes.append(frame_mae / total_count)
            binary_lengths.append(len(frame_binary))
            VBS_percentages.append(VBS_count / total_count)
            if FMEEnable:
                reference_frames[-1] = frame_interpolation(reconstructed_frame, I)
            else:
                reference_frames[-1] = reconstructed_frame
            for event in event_list:
                event.wait()
            for i in range(len(reference_frames) - 1):
                reference_frames[i] = reference_frames[-1]
        
        else:
            frame_binary += '0b0'
            for row in range(row_nums):
                if row + 2 < row_nums:
                    event_list[row + 2].wait()
                else:
                    event_list[row_nums - 1].wait()
                row_result_list = []
                # row_result_list contains:
                #   0: row_mae (float)
                #   1: row_mv_list (list of [x, y, dx, dy, refFrame])
                #   2: row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...] (I*I matrix)])
                #   3: row_binary (binary file)
                #   4: VBS_count (integer)
                inter_processing_one_row(original_second_frame, reference_frames[1: -1], row * I, I, R, QTCs[QP], QTC_subs[QP_sub], VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frame, row_result_list)
                # Load data: maes, binary_lengths, VBS_percentages, mode_mv_list, residual_list, frame_binary
                frame_mae += row_result_list[0]
                frame_binary += row_result_list[3]
                VBS_count += row_result_list[4]
                mode_mv_list += row_result_list[1]
                residual_list += row_result_list[2]
            maes.append(frame_mae / row_nums)
            binary_lengths.append(len(frame_binary))
            VBS_percentages.append(VBS_count / total_count)
            reference_frames.pop(0)
            reference_frames.pop(0)
            if FMEEnable:
                reference_frames[-1] = frame_interpolation(reconstructed_frame, I)
                reference_frames.append(np.full((2*height-2*I+1, 2*width-2*I+1, I, I), 128, dtype=np.uint8))
                reference_frames.append(np.full((2*height-2*I+1, 2*width-2*I+1, I, I), 128, dtype=np.uint8))
            else:
                reference_frames[-1] = reconstructed_frame
                reference_frames.append(np.full((height, width), 128, dtype=np.uint8))
                reference_frames.append(np.full((height, width), 128, dtype=np.uint8))
    
    if RCflag == 1:
        previous_QP = 0
        BR_table = []
        # Add frame type bit to binary and load the table.
        if is_intra_frame:  # Intra frame
            frame_binary += '0b1'
            with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Intra.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                next(reader)
                next(reader)
                BR_table = np.array([int(val) for val in next(reader)])
                
        if not is_intra_frame:  # Inter frame
            frame_binary += '0b0'
            with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Inter.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                next(reader)
                next(reader)
                BR_table = np.array([int(val) for val in next(reader)])

        # For each row, we arrange a BR for it and use the table to find related QP to encode it.
        for row in range(height // I):
            row_BR = (frame_BR - len(frame_binary)) // (height // I - row)
            # Find the QP and the QP_sub for this row.
            row_QP = 0
            if BR_table[0] > row_BR:
                QP_nums = len(BR_table)
                for i in range(QP_nums):
                    if BR_table[i] > row_BR:
                        row_QP = i + 1
                    if row_QP >= QP_nums:
                        row_QP = QP_nums - 1
            row_QP_sub = row_QP if row_QP == 0 else row_QP - 1
            row_result_list = []
            # row_result_list contains:
            # [
            #   0: row_mae (float)
            #   1: row_mode_list (list of [x, y, (0(horizontal)/1(vertical))]))
            #   2: row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...]])
            #   3: row_binary (binary file) (VBS mode and hori_verti mode are both differential, started from the first element of this row.)
            #   4: VBS_count (integer)
            # ]
            if is_intra_frame:
                intra_processing_one_row(original_second_frame, reconstructed_frame, row * I, I, QTCs[row_QP], QTC_subs[row_QP_sub], VBSEnable, Lagrange, ParallelMode, row_result_list)
            if not is_intra_frame:
                if row + 2 < row_nums:
                    event_list[row + 2].wait()
                else:
                    event_list[row_nums - 1].wait()
                inter_processing_one_row(original_second_frame, reference_frames[1: -1], row * I, I, R, QTCs[row_QP], QTC_subs[row_QP_sub], VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frame, row_result_list)

            # Load data: maes, binary_lengths, VBS_percentages, mode_mv_list, residual_list, frame_binary
            frame_mae += row_result_list[0]
            diff_QP = row_QP - previous_QP
            previous_QP = row_QP
            frame_binary += golomb_encoding([diff_QP])
            frame_binary += row_result_list[3]
            VBS_count += row_result_list[4]
            mode_mv_list += row_result_list[1]
            residual_list += row_result_list[2]
        maes.append(frame_mae / row_nums)
        binary_lengths.append(len(frame_binary))
        VBS_percentages.append(VBS_count / total_count)
        if is_intra_frame:
            if FMEEnable:
                reference_frames[-1] = frame_interpolation(reconstructed_frame, I)
            else:
                reference_frames[-1] = reconstructed_frame
            for event in event_list:
                event.wait()
            for i in range(len(reference_frames) - 1):
                reference_frames[i] = reference_frames[-1]

        if not is_intra_frame:
            reference_frames.pop(0)
            reference_frames.pop(0)
            if FMEEnable:
                reference_frames[-1] = frame_interpolation(reconstructed_frame, I)
                reference_frames.append(np.full((2*height-2*I+1, 2*width-2*I+1, I, I), 128, dtype=np.uint8))
                reference_frames.append(np.full((2*height-2*I+1, 2*width-2*I+1, I, I), 128, dtype=np.uint8))
            else:
                reference_frames[-1] = reconstructed_frame
                reference_frames.append(np.full((height, width), 128, dtype=np.uint8))
                reference_frames.append(np.full((height, width), 128, dtype=np.uint8))

    if RCflag == 2 or RCflag == 3:
        # First pass
        result_list = []
        # result_list contains:
        # 0: [row0_mae, row1_mae, row2_mae...], (float list)
        # 1: [row0_mode_list, row1_mode_list, row2_mode_list...], (list of [x, y, (0(horizontal)/1(vertical))/(dx, dy, dref_frame)])
        # 2: [row0_residual_list, row1_residual_list, row2_residual_list...], (list of [x, y [[a1,b1,c1,d1...], [a2,b2,c2,d2...]...] (I * I or (I//2 * I//2))])
        # 3: [row0_binary, row1_binary, row2_binary...], (binary list)
        # 4: [reconstructed_row0, reconstructed_row1, reconstructed_row2], (list of [[a1,b1,c1,d1...z1], [a2,b2,c2,d2...]...] (I * WIDTH))
        # 5: [row0_VBS_count, row1_VBS_count, row2_VBS_count...] (integer list)
        event_list[0].wait()
        QP = next_QP[0]
        QP_sub = QP if QP == 0 else QP - 1
        scene_change = False
        if is_intra_frame:
            intra_rows_processing(original_second_frame, I, QTCs[QP], QTC_subs[QP_sub], VBSEnable, Lagrange, ParallelMode, result_list)
        if not is_intra_frame:
            rows_mae = []
            rows_mode_mv_list = []
            rows_residual_list = []
            rows_binary = []
            rows_reconstructed = []
            rows_VBS_count = []
            for row in range(row_nums):
                if row + 2 < row_nums:
                    event_list[row + 2].wait()
                else:
                    event_list[row_nums - 1].wait()
                row_result_list = []
                # row_result_list contains:
                #   0: row_mae (float)
                #   1: row_mv_list (list of [x, y, dx, dy, refFrame])
                #   2: row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...] (I*I matrix)])
                #   3: row_binary (binary file)
                #   4: VBS_count (integer)
                inter_processing_one_row(original_second_frame, reference_frames[1: -1], row * I, I, R, QTCs[QP], QTC_subs[QP_sub], VBSEnable, Lagrange, FMEEnable, FastME, ParallelMode, reconstructed_frame, row_result_list)
                rows_mae.append(row_result_list[0])
                rows_mode_mv_list.append(row_result_list[1])
                rows_residual_list.append(row_result_list[2])
                rows_binary.append(row_result_list[3])
                rows_reconstructed.append(reconstructed_frame[row*I:(row+1)*I, :])
                rows_VBS_count.append(row_result_list[4])
            result_list.append(rows_mae)
            result_list.append(rows_mode_mv_list)
            result_list.append(rows_residual_list)
            result_list.append(rows_binary)
            result_list.append(rows_reconstructed)
            result_list.append(rows_VBS_count)
            if sum(len(binary_str) for binary_str in result_list[3]) >= scene_change_threshold(width, height, QP):
                scene_change = True

        # Second pass
        previous_QP = 0
        BR_table = []
        if is_intra_frame or scene_change:
            frame_binary += '0b1'
            with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Intra.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                next(reader)
                next(reader)
                BR_table = np.array([int(val) for val in next(reader)])
        if not (is_intra_frame or scene_change):
            frame_binary += '0b0'
            with open(f'{output_dir}/BR_tables/BR_table{table_frame_nums}_Intra.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                next(reader)
                next(reader)
                BR_table = np.array([int(val) for val in next(reader)])
        # Get the proportion table.
        proportion_table = []
        rows_BR = []
        for row_binary in result_list[3]:
            rows_BR.append(len(row_binary))
        total_BR = np.sum(rows_BR)
        for row_BR in rows_BR:
            proportion_table.append(row_BR / total_BR)
        # Get the new BR table for this frame.
        average_row_BR = total_BR / (height // I)
        scaling_factor = average_row_BR / BR_table[QP]
        for i in range(len(BR_table)):
            BR_table[i] *= scaling_factor
        # For each row, we arrange a BR for it and use the table to find related QP to encode it.
        row_QPs = []
        row_QP_subs = []
        for row in range(height // I):
            row_BR = proportion_table[row] * frame_BR
            # Find the QP and the QP_sub for this row.
            row_QP = 0
            if BR_table[0] > row_BR:
                QP_nums = len(BR_table)
                for i in range(QP_nums):
                    if BR_table[i] > row_BR:
                        row_QP = i + 1
                    if row_QP >= QP_nums:
                        row_QP = QP_nums - 1
            row_QPs.append(row_QP)
            row_QP_sub = row_QP if row_QP == 0 else row_QP - 1
            row_QP_subs.append(row_QP_sub)

        # update next QP value
        next_QP[0] = int(np.mean(row_QPs))

        for row in range(height // I):
            row_QP = row_QPs[row]
            row_QP_sub = row_QP_subs[row]
            # encode this row with the row_QP.
            row_result_list = []
            # row_result_list contains:
            # 0: row_mae (float)
            # 1: row_mode_list (list of [x, y, (0(horizontal)/1(vertical))]) / (list of [x, y, dx, dy, refFrame])
            # 2: row_residual_list (list of [x, y, [[a1, b1, c1, d1...], [a2, b2, c2, d2...]...]])
            # 3: row_binary (binary file)
            # 4: VBS_count (integer)
            if (not is_intra_frame) and row_QP == QP and (not scene_change) and RCflag == 3: # Can benefit from first pass
                row_result_list.append(result_list[0][row])
                row_result_list.append(result_list[1][row])
                row_result_list.append(result_list[2][row])
                row_result_list.append(result_list[3][row])
                row_result_list.append(result_list[5][row])
                reconstructed_frame[row*I:(row+1)*I, :] = result_list[4][row]
            if not ((not is_intra_frame) and row_QP == QP and (not scene_change) and RCflag == 3):
                if is_intra_frame or scene_change:
                    intra_processing_one_row(original_second_frame, reconstructed_frame, row * I, I, QTCs[row_QP], QTC_subs[row_QP_sub], VBSEnable, Lagrange, ParallelMode, row_result_list)
                    
                if not (is_intra_frame or scene_change):
                    inter_processing_one_row(original_second_frame, reference_frames[1: -1], row * I, I, R, QTCs[row_QP], QTC_subs[row_QP_sub], VBSEnable, Lagrange2, FMEEnable, FastME, ParallelMode, reconstructed_frame, row_result_list)

            # Load data: maes, binary_lengths, VBS_percentages, mode_mv_list, residual_list, frame_binary
            frame_mae += row_result_list[0]
            diff_QP = row_QP - previous_QP
            previous_QP = row_QP
            frame_binary += golomb_encoding([diff_QP]) + row_result_list[3]
            VBS_count += row_result_list[4]
            mode_mv_list += row_result_list[1]
            residual_list += row_result_list[2]
        maes.append(frame_mae / row_nums)
        binary_lengths.append(len(frame_binary))
        VBS_percentages.append(VBS_count / total_count)

        if is_intra_frame or scene_change:
            if FMEEnable:
                reference_frames[-1] = frame_interpolation(reconstructed_frame, I)
            else:
                reference_frames[-1] = reconstructed_frame
            for event in event_list:
                event.wait()
            for i in range(len(reference_frames) - 1):
                reference_frames[i] = reference_frames[-1]

        if not (is_intra_frame or scene_change):
            reference_frames.pop(0)
            reference_frames.pop(0)
            if FMEEnable:
                reference_frames[-1] = frame_interpolation(reconstructed_frame, I)
                reference_frames.append(np.full((2*height-2*I+1, 2*width-2*I+1, I, I), 128, dtype=np.uint8))
                reference_frames.append(np.full((2*height-2*I+1, 2*width-2*I+1, I, I), 128, dtype=np.uint8))
            else:
                reference_frames[-1] = reconstructed_frame
                reference_frames.append(np.full((height, width), 128, dtype=np.uint8))
                reference_frames.append(np.full((height, width), 128, dtype=np.uint8))

    if len(mode_mv_list) > 0 and len(mode_mv_list[0]) == 3:
        # Save mode to txt
            mode_file_path = os.path.join(output_dir, 'mode_and_motion_vector', f'frame_{frame_num}_mode.txt')
            os.makedirs(os.path.dirname(mode_file_path), exist_ok=True)
            with open(mode_file_path, 'w') as f:
                for val in mode_mv_list:
                    f.write(f"{val[0]} {val[1]} {val[2]}\n")
    if len(mode_mv_list) > 0 and len(mode_mv_list[0]) == 5:
        # Save mode and motion vectors vectors to txt
            mv_file_path = os.path.join(output_dir, 'mode_and_motion_vector', f'frame_{frame_num}_motion_vector.txt')
            os.makedirs(os.path.dirname(mv_file_path), exist_ok=True)
            with open(mv_file_path, 'w') as f:
                for val in mode_mv_list:
                    f.write(f"{val[0]} {val[1]} {val[2]} {val[3]} {val[4]}\n")  # Include reference frame index

    # Save the reconstructed frame
    reconstructed_file_path = os.path.join(output_dir, 'y_frames_reconstructed', f'y_reconstructed_frame_{frame_num}.y')
    os.makedirs(os.path.dirname(reconstructed_file_path), exist_ok=True)
    with open(reconstructed_file_path, 'wb') as f:
        f.write(reconstructed_frame[:height, :width].tobytes())
    
    # Save the residual values
    residuals_file_path = os.path.join(output_dir, 'residuals', f'residuals_frame_{frame_num}.txt')
    os.makedirs(os.path.dirname(residuals_file_path), exist_ok=True)
    with open(residuals_file_path, 'w') as f:
        for res in residual_list:
            residual_str = json.dumps(res[2].tolist())
            f.write(f"{res[0]} {res[1]} {residual_str}\n")
    
    # Save the encoded binary
    binary_file_path = os.path.join(output_dir, 'encoded_binary', f'binary_frame_{frame_num}')
    os.makedirs(os.path.dirname(binary_file_path), exist_ok=True)
    with open(binary_file_path, 'wb') as f:
        frame_binary.tofile(f)

#   parallel_frame_encode function CONSUMES:
# file_dir: directory for the original y-frames.
# frame_nums: number of the frames to encode.
# WIDTH
# HEIGHT
# I: block size
# R: search range
# QP: quantization parameter
# QTCs: a list of QTCs with block size I.
# QTC_subs: a list of QTCs with block size I // 2.
# I_PERIOD: the period of i frames. (0, i-1, 2i-1, 3i-1) are all intra frames.
# OUTPUT_DIR
# reference_frames: reconstructed frames.
# VBSEnable
# FMEEnable
# FastME
# Lagrange: intra lagrange
# Lagrange2: inter lagrange
# RCflag
# frame_BR: bit rate for this frame.
# ParallelMode
#   parallel_frame_encode function PRODUCES:
# maes: A list of all frames' Mean of Absolute Error
# binary_lengths: A list of all frames' binary file length
# encoding_time: total encoding time. (in sec)
# VBS_percentages: A list of all frames' VBS percentages
#   parallel_frame_encode function EFFECTS:
# saves files: frame_x_mode/motion_vector.txt; reconstructed_frame; residual frame; binary frame.
def parallel_frame_encode(file_dir, frame_nums, table_frame_nums, width, height, I, R, QP, QTCs, QTC_subs, I_period, output_dir, nRefFrames, VBSEnable, FMEEnable, FastME, Lagrange, Lagrange2, RCflag, frame_BR, ParallelMode):
    start_time = time.time()
    maes = []
    binary_lengths = []
    VBS_percentages = []
    reference_frames = []
    next_QP = [QP]
    if FMEEnable:
        reference_frames = [np.full((2*height-2*I+1, 2*width-2*I+1, I, I), 128, dtype=np.uint8) for _ in range(nRefFrames + 2)]
        R *= 2
    if not FMEEnable:
        reference_frames = [np.full((height, width), 128, dtype=np.uint8) for _ in range(nRefFrames + 2)]
    for frame_num in tqdm(range(0, frame_nums, 2)):
        if frame_num + 2 > frame_nums:
            reference_frames = reference_frames[:-2]
            mae, b_len, encoding_t, reference_frames, VBS_percentage, QP = encode_frame(file_dir, frame_num, table_frame_nums, width, height, I, R, QP, QTCs, QTC_subs, I_period, output_dir, reference_frames, VBSEnable, FMEEnable, FastME, Lagrange, Lagrange2, RCflag, frame_BR, ParallelMode)
            maes.append(mae)
            binary_lengths.append(b_len)
            VBS_percentages.append(VBS_percentage)
            break
        event_list = [threading.Event() for _ in range(height // I)]
        original_first_frame = read_and_pad(f"{file_dir}/y_frame_{frame_num}.y", I, width, height)
        first_thread = threading.Thread(target=first_frame,
                                        args=(original_first_frame, frame_num, table_frame_nums, width, height, I, R, next_QP, QTCs, QTC_subs, I_period, output_dir, reference_frames, VBSEnable, FMEEnable, FastME, Lagrange, Lagrange2, RCflag, frame_BR, ParallelMode, event_list, maes, binary_lengths, VBS_percentages))
        first_thread.start()
        frame_num += 1
        original_second_frame = read_and_pad(f"{file_dir}/y_frame_{frame_num}.y", I, width, height)
        second_thread = threading.Thread(target=second_frame,
                                         args=(original_second_frame, frame_num, table_frame_nums, width, height, I, R, next_QP, QTCs, QTC_subs, I_period, output_dir, reference_frames, VBSEnable, FMEEnable, FastME, Lagrange, Lagrange2, RCflag, frame_BR, ParallelMode, event_list, maes, binary_lengths, VBS_percentages))
        second_thread.start()
        first_thread.join()
        second_thread.join()
    end_time = time.time()
    return maes, binary_lengths, end_time - start_time, VBS_percentages
