import os
import numpy as np
import time

from util import *


def decode_header(OUTPUT_DIR):
    with open(f"{OUTPUT_DIR}/encoded_binary/header", 'rb') as f:
        header = golomb_decoding(Bits(f))
    
    return header

def intra_full_block_decode(frame_sequence, decoded_frame, previous_mode, i, I, QTC, x_idx, y_idx, x, y):
    current_mode = previous_mode + frame_sequence[i]
    i += 1
    skip_i, RLE_decoded_list = run_length_decoding(frame_sequence[i:], I)
    i += skip_i
    trans_quantized_block = reverse_diagonal(RLE_decoded_list, I, x_idx, y_idx)
    residual_block = inverse_quantization(trans_quantized_block, QTC, I)
    residual_block = IDCT(residual_block)
    predictor_block = intra_return_predictor(decoded_frame, current_mode, x, y, I)
    decoded_frame[y:y+I, x:x+I] = add_with_saturation(predictor_block, residual_block)
    
    # print(x,y, trans_quantized_block)
    # print(x,y, current_mode, predictor_block)
    
    return i, current_mode
    
def intra_sub_block_decode(frame_sequence, decoded_frame, previous_mode, i, I, QTC, x_idx, y_idx, x, y):
    # x, y
    current_mode1 = previous_mode + frame_sequence[i]
    i += 1
    skip_i, RLE_decoded_list = run_length_decoding(frame_sequence[i:], I)
    i += skip_i
    trans_quantized_block = reverse_diagonal(RLE_decoded_list, I, x_idx, y_idx)
    residual_block = inverse_quantization(trans_quantized_block, QTC, I)
    residual_block = IDCT(residual_block)
    predictor_block = intra_return_predictor(decoded_frame, current_mode1, x, y, I)
    decoded_frame[y:y+I, x:x+I] = add_with_saturation(predictor_block, residual_block)
    
    # print(trans_quantized_block)
    # print(x, y, current_mode1, predictor_block)
    # print(x, y, residual_block)
    
    # x+I, y
    current_mode2 = current_mode1 + frame_sequence[i]
    i += 1
    skip_i, RLE_decoded_list = run_length_decoding(frame_sequence[i:], I)
    i += skip_i
    trans_quantized_block = reverse_diagonal(RLE_decoded_list, I, x_idx, y_idx)
    residual_block = inverse_quantization(trans_quantized_block, QTC, I)
    residual_block = IDCT(residual_block)
    predictor_block = intra_return_predictor(decoded_frame, current_mode2, x+I, y, I)
    decoded_frame[y:y+I, x+I:x+I+I] = add_with_saturation(predictor_block, residual_block)
    
    # print(trans_quantized_block)
    # print(x+I, y, current_mode2, predictor_block)
    # print(x+I, y, residual_block)
    
    # x, y+I
    current_mode3 = current_mode2 + frame_sequence[i]
    i += 1
    skip_i, RLE_decoded_list = run_length_decoding(frame_sequence[i:], I)
    i += skip_i
    trans_quantized_block = reverse_diagonal(RLE_decoded_list, I, x_idx, y_idx)
    residual_block = inverse_quantization(trans_quantized_block, QTC, I)
    residual_block = IDCT(residual_block)
    predictor_block = intra_return_predictor(decoded_frame, current_mode3, x, y+I, I)
    decoded_frame[y+I:y+I+I, x:x+I] = add_with_saturation(predictor_block, residual_block)
    
    # print(trans_quantized_block)
    # print(x, y+I, current_mode3, predictor_block)
    # print(x, y+I, residual_block)
    
    # x+I, y+I
    current_mode4 = current_mode3 + frame_sequence[i]
    i += 1
    skip_i, RLE_decoded_list = run_length_decoding(frame_sequence[i:], I)
    i += skip_i
    trans_quantized_block = reverse_diagonal(RLE_decoded_list, I, x_idx, y_idx)
    residual_block = inverse_quantization(trans_quantized_block, QTC, I)
    residual_block = IDCT(residual_block)
    predictor_block = intra_return_predictor(decoded_frame, current_mode4, x+I, y+I, I)
    decoded_frame[y+I:y+I+I, x+I:x+I+I] = add_with_saturation(predictor_block, residual_block)
    
    # print(trans_quantized_block)
    # print(x+I, y+I, current_mode4, predictor_block)
    # print(x+I, y+I, residual_block)
    
    return i, current_mode4

def decode_intra(frame_binary, WIDTH, HEIGHT, I, QP, QTCs, QTCs_sub, x_idx, y_idx, xsub_idx, ysub_idx, VBSEnable, RCflag, ParallelMode):
    decoded_frame = np.zeros((HEIGHT*2, WIDTH*2), dtype=np.uint8) # allocate more in case of padding
    frame_sequence = golomb_decoding(frame_binary)
    previous_mode = 0
    current_QP = QP
    if(RCflag>=1):
        current_QP = frame_sequence[0]
        frame_sequence = frame_sequence[1:]
    QP_sub = current_QP if current_QP == 0 else current_QP - 1

    i = 0
    y = 0
    x = 0
    
    # total_count = 0
    # sub_count = 0
    
    while i < len(frame_sequence):
        if(VBSEnable):
            if(frame_sequence[i]): # split
                i = i+1
                i, current_mode = intra_sub_block_decode(frame_sequence, decoded_frame, previous_mode, i, I//2, QTCs_sub[QP_sub], xsub_idx, ysub_idx, x, y)
                # sub_count += 1
                
            else: # no split
                i = i+1
                i, current_mode = intra_full_block_decode(frame_sequence, decoded_frame, previous_mode, i, I, QTCs[current_QP], x_idx, y_idx, x, y)
            
        else:
            i, current_mode = intra_full_block_decode(frame_sequence, decoded_frame, previous_mode, i, I, QTCs[current_QP], x_idx, y_idx, x, y)
            
        previous_mode = 0 if ParallelMode == 1 else current_mode # ParallelMode = 1 doesn't use diff encoding
        
        # total_count += 1
        x += I
        if(x >= WIDTH):
            x = 0
            y += I
            previous_mode = 0
            if(RCflag>=1 and i < len(frame_sequence)):
                previous_QP = current_QP
                current_QP = frame_sequence[i] + previous_QP
                QP_sub = current_QP if current_QP == 0 else current_QP - 1
                i += 1
            
    return decoded_frame[:HEIGHT, :WIDTH]

def inter_full_block_decode(frame_sequence, reference_frames, decoded_frame, previous_mv, i, I, QTC, x_idx, y_idx, x, y, FMEEnable):
    current_mv_x = previous_mv[0] + frame_sequence[i]
    current_mv_y = previous_mv[1] + frame_sequence[i+1]
    current_ref_idx = previous_mv[2] + frame_sequence[i+2]
    i += 3
    skip_i, RLE_decoded_list = run_length_decoding(frame_sequence[i:], I)
    i += skip_i
    trans_quantized_block = reverse_diagonal(RLE_decoded_list, I, x_idx, y_idx)
    residual_block = inverse_quantization(trans_quantized_block, QTC, I)
    residual_block = IDCT(residual_block)
    
    # print(x, y, current_mv_x, current_mv_y)
    
    if(FMEEnable):
        predicted_block = reference_frames[current_ref_idx][y*2+current_mv_y, x*2+current_mv_x]
    else:
        predicted_block = reference_frames[current_ref_idx][y+current_mv_y:y+current_mv_y+I, x+current_mv_x:x+current_mv_x+I]
    
    decoded_frame[y:y+I, x:x+I] = add_with_saturation(predicted_block, residual_block)
    return i, current_mv_x, current_mv_y, current_ref_idx

def inter_sub_block_decode(frame_sequence, reference_frames, decoded_frame, previous_mv, i, I, QTC, x_idx, y_idx, x, y, FMEEnable):
    current_mv_x1 = previous_mv[0] + frame_sequence[i]
    current_mv_y1 = previous_mv[1] + frame_sequence[i+1]
    current_ref_idx1 = previous_mv[2] + frame_sequence[i+2]
    i += 3
    skip_i, RLE_decoded_list = run_length_decoding(frame_sequence[i:], I)
    i += skip_i
    trans_quantized_block = reverse_diagonal(RLE_decoded_list, I, x_idx, y_idx)
    residual_block = inverse_quantization(trans_quantized_block, QTC, I)
    residual_block = IDCT(residual_block)
    if(FMEEnable):
        predicted_block = reference_frames[current_ref_idx1][y*2+current_mv_y1, x*2+current_mv_x1, 0:I, 0:I]
    else:
        predicted_block = reference_frames[current_ref_idx1][y+current_mv_y1:y+current_mv_y1+I, x+current_mv_x1:x+current_mv_x1+I]
    decoded_frame[y:y+I, x:x+I] = add_with_saturation(predicted_block, residual_block)
    
    # print(x,y, current_mv_x1, current_mv_y1, current_ref_idx1)
    
    # x+I, y
    current_mv_x2 = current_mv_x1 + frame_sequence[i]
    current_mv_y2 = current_mv_y1 + frame_sequence[i+1]
    current_ref_idx2 = current_ref_idx1 + frame_sequence[i+2]
    # print(x+I,y, current_mv_x2, current_mv_y2, current_ref_idx2)
    i += 3
    skip_i, RLE_decoded_list = run_length_decoding(frame_sequence[i:], I)
    i += skip_i
    trans_quantized_block = reverse_diagonal(RLE_decoded_list, I, x_idx, y_idx)
    residual_block = inverse_quantization(trans_quantized_block, QTC, I)
    residual_block = IDCT(residual_block)
    if(FMEEnable):
        predicted_block = reference_frames[current_ref_idx2][y*2+current_mv_y2, x*2+current_mv_x2, 0:I, I:I+I]
    else:
        predicted_block = reference_frames[current_ref_idx2][y+current_mv_y2:y+current_mv_y2+I, x+current_mv_x2+I:x+current_mv_x2+I+I]
    decoded_frame[y:y+I, x+I:x+I+I] = add_with_saturation(predicted_block, residual_block)
    
    # print(x+I,y, current_mv_x2, current_mv_y2, current_ref_idx2)
    
    # x, y+I
    current_mv_x3 = current_mv_x2 + frame_sequence[i]
    current_mv_y3 = current_mv_y2 + frame_sequence[i+1]
    current_ref_idx3 = current_ref_idx2 + frame_sequence[i+2]
    i += 3
    skip_i, RLE_decoded_list = run_length_decoding(frame_sequence[i:], I)
    i += skip_i
    trans_quantized_block = reverse_diagonal(RLE_decoded_list, I, x_idx, y_idx)
    residual_block = inverse_quantization(trans_quantized_block, QTC, I)
    residual_block = IDCT(residual_block)
    if(FMEEnable):
        predicted_block = reference_frames[current_ref_idx3][y*2+current_mv_y3, x*2+current_mv_x3, I:I+I, 0:I]
    else:
        predicted_block = reference_frames[current_ref_idx3][y+current_mv_y3+I:y+current_mv_y3+I+I, x+current_mv_x3:x+current_mv_x3+I]
    decoded_frame[y+I:y+I+I, x:x+I] = add_with_saturation(predicted_block, residual_block)
    
    # x+I, y+I
    current_mv_x4 = current_mv_x3 + frame_sequence[i]
    current_mv_y4 = current_mv_y3 + frame_sequence[i+1]
    current_ref_idx4 = current_ref_idx3 + frame_sequence[i+2]
    i += 3
    skip_i, RLE_decoded_list = run_length_decoding(frame_sequence[i:], I)
    i += skip_i
    trans_quantized_block = reverse_diagonal(RLE_decoded_list, I, x_idx, y_idx)
    residual_block = inverse_quantization(trans_quantized_block, QTC, I)
    residual_block = IDCT(residual_block)
    if(FMEEnable):
        predicted_block = reference_frames[current_ref_idx4][y*2+current_mv_y4, x*2+current_mv_x4, I:I+I, I:I+I]
    else:
        predicted_block = reference_frames[current_ref_idx4][y+current_mv_y4+I:y+current_mv_y4+I+I, x+current_mv_x4+I:x+current_mv_x4+I+I]
    decoded_frame[y+I:y+I+I, x+I:x+I+I] = add_with_saturation(predicted_block, residual_block)
    return i, current_mv_x4, current_mv_y4, current_ref_idx4

def decode_inter(frame_binary, reference_frames, WIDTH, HEIGHT, I, QP, QTCs, QTCs_sub, x_idx, y_idx, xsub_idx, ysub_idx, VBSEnable, FMEEnable, RCflag, ParallelMode):
    decoded_frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    frame_sequence = golomb_decoding(frame_binary)
    previous_mv = [0,0,0]
    current_QP = QP
    if(RCflag>=1):
        current_QP = frame_sequence[0]
        frame_sequence = frame_sequence[1:]
    QP_sub = current_QP if current_QP == 0 else current_QP - 1
    
    i = 0
    y = 0
    x = 0
    
    # total_count = 0
    # sub_count = 0

    while i < len(frame_sequence):
        if(VBSEnable):
            if(frame_sequence[i]):
                i += 1
                i, current_mv_x, current_mv_y, current_ref_idx = inter_sub_block_decode(frame_sequence, reference_frames, decoded_frame, previous_mv, i, I//2, QTCs_sub[QP_sub], xsub_idx, ysub_idx, x, y, FMEEnable)
                # sub_count += 1
            else:
                i += 1
                i, current_mv_x, current_mv_y, current_ref_idx = inter_full_block_decode(frame_sequence, reference_frames, decoded_frame, previous_mv, i, I, QTCs[current_QP], x_idx, y_idx, x, y, FMEEnable)
        else:
            i, current_mv_x, current_mv_y, current_ref_idx = inter_full_block_decode(frame_sequence, reference_frames, decoded_frame, previous_mv, i, I, QTCs[current_QP], x_idx, y_idx, x, y, FMEEnable)
        
        previous_mv[0] = 0 if ParallelMode == 1 else current_mv_x # ParallelMode = 1 doesn't use diff encoding
        previous_mv[1] = 0 if ParallelMode == 1 else current_mv_y
        previous_mv[2] = 0 if ParallelMode == 1 else current_ref_idx
        
        # total_count += 1
        x += I
        if(x >= WIDTH):
            x = 0
            y += I
            previous_mv = [0,0,0]
            if(RCflag>=1 and i < len(frame_sequence)):
                previous_QP = current_QP
                current_QP = frame_sequence[i] + previous_QP
                QP_sub = current_QP if current_QP == 0 else current_QP - 1
                i += 1
            
    return decoded_frame[:HEIGHT, :WIDTH]


def decode_frame(frame_num, WIDTH, HEIGHT, I, QP, QTCs, QTCs_sub, x_idx, y_idx, xsub_idx, ysub_idx, OUTPUT_DIR, reference_frames, VBSEnable, FMEEnable,nRefFrames, RCflag, ParallelMode):
    start_time = time.time()  # Start the timer
    # Read the binary file for the current frame
    with open(f"{OUTPUT_DIR}/encoded_binary/binary_frame_{frame_num}", 'rb') as f:
        frame_binary = Bits(f)

    if frame_binary[0]:  # Intra
        decoded_frame = decode_intra(frame_binary[1:], WIDTH, HEIGHT, I, QP, QTCs, QTCs_sub, x_idx, y_idx, xsub_idx, ysub_idx, VBSEnable, RCflag, ParallelMode)
        if(FMEEnable):
            FME_decoded_frame = frame_interpolation(decoded_frame, I)
            reference_frames = [FME_decoded_frame for _ in range(nRefFrames)]
        else:
            reference_frames = [decoded_frame for _ in range(len(reference_frames))]
            
    else:  # Inter
        # Decode the inter frame
        decoded_frame = decode_inter(frame_binary[1:], reference_frames, WIDTH, HEIGHT, I, QP, QTCs, QTCs_sub, x_idx, y_idx, xsub_idx, ysub_idx, VBSEnable, FMEEnable, RCflag, ParallelMode)
        # Update the reference frames list
        reference_frames.pop(0)  # Remove the oldest reference frame
        if(FMEEnable):
            reference_frames.append(frame_interpolation(decoded_frame, I))
        else:
            reference_frames.append(decoded_frame)  # Add the new decoded frame

    # Save the decoded frame to the output directory
    if not os.path.exists(f"{OUTPUT_DIR}/y_frames_decoded"):
        os.makedirs(f"{OUTPUT_DIR}/y_frames_decoded")
    with open(f"{OUTPUT_DIR}/y_frames_decoded/y_decoded_frame_{frame_num}.y", 'wb') as f:
        f.write(decoded_frame[:HEIGHT, :WIDTH].tobytes())
    decoding_time = time.time() - start_time
    return decoding_time, reference_frames