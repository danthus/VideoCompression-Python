import os
import numpy as np
import threading

from scipy.fftpack import dctn, idctn
from bitstring import Bits


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))
    
def read_and_pad(filename,I,WIDTH,HEIGHT):
    
    # Load the Y-only frame data
    with open(filename, 'rb') as f:
        frame_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(HEIGHT, WIDTH)

    # Determine the padding needed
    pad_width = (I - (WIDTH % I)) % I
    pad_height = (I - (HEIGHT % I)) % I

    # Apply the padding to the frame
    padded_frame = np.pad(frame_data, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=128)
    
    return padded_frame

def find_best_block(padded_frame, previous_frame, x, y,I,R,WIDTH,HEIGHT):
    current_block = padded_frame[y:y+I, x:x+I]
    min_mae = np.mean(np.abs(current_block.astype(np.int16) - previous_frame[y:y+I, x:x+I].astype(np.int16)))
    best_mv = (0, 0)

    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if x + dx < 0 or x + dx + I > WIDTH or y + dy < 0 or y + dy + I > HEIGHT:
                continue  # skip out-of-bound blocks
            
            ref_block = previous_frame[y+dy:y+dy+I, x+dx:x+dx+I]
            mae = np.mean(np.abs(current_block.astype(np.int16) - ref_block.astype(np.int16)))
            if mae < min_mae:
                min_mae = mae
                best_mv = (dx, dy)
                
    return best_mv, min_mae

def find_best_block_FME(padded_frame, previous_frame, x, y,I,R, pos):
    FME_x = x*2
    FME_y = y*2
    current_block = padded_frame[y:y+I, x:x+I]
    if(pos == 0):
        min_mae = np.mean(np.abs(current_block.astype(np.int16) - previous_frame[FME_y, FME_x].astype(np.int16)))
    elif(pos == 1):
        min_mae = np.mean(np.abs(current_block.astype(np.int16) - previous_frame[FME_y, FME_x, 0:I, 0:I].astype(np.int16)))
    elif(pos == 2):
        min_mae = np.mean(np.abs(current_block.astype(np.int16) - previous_frame[FME_y, FME_x, 0:I, I:I+I].astype(np.int16)))
    elif(pos == 3):
        min_mae = np.mean(np.abs(current_block.astype(np.int16) - previous_frame[FME_y, FME_x, I:I+I, 0:I].astype(np.int16)))
    elif(pos == 4):
        min_mae = np.mean(np.abs(current_block.astype(np.int16) - previous_frame[FME_y, FME_x, I:I+I, I:I+I].astype(np.int16)))
    best_mv = (0, 0)
    
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if FME_y + dy < 0 or FME_y + dy >= previous_frame.shape[0] or FME_x + dx < 0 or FME_x + dx >= previous_frame.shape[1]:
                continue  # skip out-of-bound blocks
            
            if(pos == 0):
                ref_block = previous_frame[FME_y+dy, FME_x+dx]
            elif(pos == 1):
                ref_block = previous_frame[FME_y+dy, FME_x+dx][0:I, 0:I]
            elif(pos == 2):
                ref_block = previous_frame[FME_y+dy, FME_x+dx][0:I, I:I+I]
            elif(pos == 3):
                ref_block = previous_frame[FME_y+dy, FME_x+dx][I:I+I, 0:I]
            elif(pos == 4):
                ref_block = previous_frame[FME_y+dy, FME_x+dx][I:I+I, I:I+I]
            
            mae = np.mean(np.abs(current_block.astype(np.int16) - ref_block.astype(np.int16)))
            
            if mae < min_mae:
                min_mae = mae
                best_mv = (dx, dy)
                
    return best_mv, min_mae

def find_best_block_fastME(padded_frame, previous_frame, x, y, I, R, WIDTH, HEIGHT, MVP):
    # Fast Motion Estimation algorithm
    rad = 0
    search_pattern = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    current_block = padded_frame[y:y + I, x:x + I]
    min_mae = np.mean(np.abs(current_block.astype(np.int16) - previous_frame[y:y+I, x:x+I].astype(np.int16)))
    best_mv = (0, 0)

    while rad < R:
        for dx, dy in search_pattern:
            mvx, mvy = MVP[0] + dx, MVP[1] + dy
            if x + mvx < 0 or x + mvx + I > WIDTH or y + mvy < 0 or y + mvy + I > HEIGHT:
                continue  # Skip out-of-bound blocks

            ref_block = previous_frame[y + mvy:y + mvy + I, x + mvx:x + mvx + I]
            mae = np.mean(np.abs(current_block.astype(np.int16) - ref_block.astype(np.int16)))

            if mae < min_mae:
                min_mae = mae
                best_mv = (mvx, mvy)
        # update MVP if not on center 
        if best_mv == MVP:
            break
        else:
            MVP = best_mv

        rad += 1
    return best_mv, min_mae

def find_best_block_fastME_FME(padded_frame, previous_frame, x, y, I, R, WIDTH, HEIGHT, MVP, pos):
    rad = 0
    FME_x = x*2
    FME_y = y*2
    search_pattern = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    current_block = padded_frame[y:y + I, x:x + I]
    if(pos == 0):
        min_mae = np.mean(np.abs(current_block.astype(np.int16) - previous_frame[FME_y, FME_x].astype(np.int16)))
    elif(pos == 1):
        min_mae = np.mean(np.abs(current_block.astype(np.int16) - previous_frame[FME_y, FME_x, 0:I, 0:I].astype(np.int16)))
    elif(pos == 2):
        min_mae = np.mean(np.abs(current_block.astype(np.int16) - previous_frame[FME_y, FME_x, 0:I, I:I+I].astype(np.int16)))
    elif(pos == 3):
        min_mae = np.mean(np.abs(current_block.astype(np.int16) - previous_frame[FME_y, FME_x, I:I+I, 0:I].astype(np.int16)))
    elif(pos == 4):
        min_mae = np.mean(np.abs(current_block.astype(np.int16) - previous_frame[FME_y, FME_x, I:I+I, I:I+I].astype(np.int16)))
    best_mv = (0, 0)

    while rad < R:
        for dx, dy in search_pattern:
            mvx, mvy = MVP[0] + dx, MVP[1] + dy
            if FME_y + mvy < 0 or FME_y + mvy >= previous_frame.shape[0] or FME_x + mvx < 0 or FME_x + mvx >= previous_frame.shape[1]:
                continue  # Skip out-of-bound blocks
            
            if(pos == 0):
                ref_block = previous_frame[FME_y+mvy, FME_x+mvx]
            elif(pos == 1):
                ref_block = previous_frame[FME_y+mvy, FME_x+mvx][0:I, 0:I]
            elif(pos == 2):
                ref_block = previous_frame[FME_y+mvy, FME_x+mvx][0:I, I:I+I]
            elif(pos == 3):
                ref_block = previous_frame[FME_y+mvy, FME_x+mvx][I:I+I, 0:I]
            elif(pos == 4):
                ref_block = previous_frame[FME_y+mvy, FME_x+mvx][I:I+I, I:I+I]
                
            mae = np.mean(np.abs(current_block.astype(np.int16) - ref_block.astype(np.int16)))

            if mae < min_mae:
                min_mae = mae
                best_mv = (mvx, mvy)
                
        # update MVP if not on center 
        if best_mv == MVP:
            break
        else:
            MVP = best_mv

        rad += 1
        
    return best_mv, min_mae

def zigzag(block):
    return np.concatenate([np.diagonal(block[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-block.shape[0], block.shape[0])])

def diagonal(block):
    return np.concatenate([np.fliplr(block).diagonal(k) for k in range(int(block.shape[0] - 1), int(-block.shape[0]), -1)])

def combination(val):
    return tuple(y for y in range(val)), tuple(x for x in range(val-1, -1, -1))

def create_reverse_diagonal_idx(I):
    y_idx = tuple()
    x_idx = tuple()
            
    for i in range(1, I+1, 1):
        y_idx += tuple(idx for idx in range(i))
    for i in range(-I+1, 0, 1):
        y_idx += y_idx[i:]
    
    for i in range(I):
        x_idx += tuple(idx for idx in range(i, -1, -1))
    for i in range(-I, -1, 1):
        x_idx += x_idx[i:-1]
    
    return x_idx, y_idx

def reverse_diagonal(arr, I, x_idx, y_idx):
    block = np.zeros((I,I))
    block[y_idx, x_idx] = arr[:]
    return block   

def DCT(block):
    return np.rint(dctn(block, type=2 ,norm='ortho')).astype(np.int16)

def IDCT(block):
    return np.rint(idctn(block, type=2 ,norm='ortho')).astype(np.int16)

# quantization transform coefficients
# QP is between 0 and log2(i) + 7
# QP = 0 means best quality
def create_QTC(I, QP):
    if( QP < 0 or QP > np.log2(I)+7 ):
        print("invalid QP value")
        exit()
    
    QTC = np.zeros((I,I))
    for y in range(I):
        for x in range(I):
            if( x+y < I-1):
                QTC[y][x] = int(round(2**QP))
            elif (x+y == I-1):
                QTC[y][x] = int(round(2**(QP+1)))
            else:
                QTC[y][x] = int(round(2**(QP+2)))
    
    return QTC

def create_all_QTC(I):
    max_QP = int(np.log2(I)+7)
    QTCs = np.zeros((max_QP+1, I, I))
    for QP in range(max_QP+1):
        QTCs[QP] = create_QTC(I, QP)
        
    return QTCs

def quantization(coeff, QTC, I):
    return np.rint(coeff/QTC).astype(np.int16)

def inverse_quantization(quantized_coeff, QTC, I):
    return np.rint(quantized_coeff*QTC).astype(np.int16)

def intra_find_best_predictor(frame, reconstructed_frame, x, y, I):
    current_block = frame[y:y+I, x:x+I]
    if(x == 0):
        horizontal_predictor = np.full((I,I), 128, dtype=np.int16)
    else:
        horizontal_predictor = np.reshape(np.repeat(reconstructed_frame[y:y+I, x-1], I), (I,I))
        
    if(y == 0):
        vertical_predictor = np.full((I,I), 128, dtype=np.int16)
    else:
        vertical_predictor = np.reshape(np.repeat(reconstructed_frame[y-1, x:x+I], I), (I,I), order='F')
    
    h_residual = current_block.astype(np.int16) - horizontal_predictor.astype(np.int16)
    v_residual = current_block.astype(np.int16) - vertical_predictor.astype(np.int16)
    h_mae = np.mean(np.abs(h_residual))
    v_mae = np.mean(np.abs(v_residual))
    
    if h_mae <= v_mae:
        return 0, h_mae, h_residual, horizontal_predictor
    else:
        return 1, v_mae, v_residual, vertical_predictor
        
def intra_return_predictor(frame, mode, x, y, I):
    if(mode): # vertical
        if(y == 0):
            predictor = np.full((I,I), 128, dtype=np.int16)
        else:
            predictor = np.reshape(np.repeat(frame[y-1, x:x+I], I), (I,I), order='F')
    else: # horizontal
        if(x == 0):
            predictor = np.full((I,I), 128, dtype=np.int16)
        else:
            predictor = np.reshape(np.repeat(frame[y:y+I, x-1], I), (I,I))
            
    return predictor.astype(np.int16)

def frame_interpolation(frame, I):
    H = frame.shape[0]
    W = frame.shape[1]
    
    FME_frame = np.zeros((2*H-2*I+1, 2*W-2*I+1, I, I), dtype=np.int16)

    # put frame into FME frame
    for y in range(H-I+1):
        for x in range(W-I+1):
            FME_frame[y*2, x*2] = frame[y:y+I, x:x+I]
            
    # interpolate (odd,even) and (even, odd) positions
    for y in range(FME_frame.shape[0]):
        for x in range(FME_frame.shape[1]):
            if (y%2 == 0 and x%2 == 1):
                # horizontal case
                FME_frame[y,x] = np.rint((FME_frame[y, x-1]+ FME_frame[y, x+1])/2)
            elif (y%2 == 1 and x%2 == 0):
                # vertical case
                FME_frame[y,x] = np.rint((FME_frame[y-1, x] + FME_frame[y+1, x])/2)
                
    # interpolate (odd,odd) positions
    for y in range(FME_frame.shape[0]):
        for x in range(FME_frame.shape[1]):            
                if(y%2 == 1 and x%2 == 1):
                    FME_frame[y,x] = np.rint((FME_frame[y, x-1] + FME_frame[y, x+1] + FME_frame[y-1, x] + FME_frame[y+1, x])/4)
                
    return FME_frame.astype(np.uint8)

def run_length_encoding(data_list):
    encoded_list = []
    idx1 = 0
    idx2 = 0
    while(idx2 < len(data_list)):
        if(data_list[idx2] == 0 and data_list[idx1] != 0):
            encoded_list.append(idx1-idx2)
            encoded_list.extend(data_list[idx1:idx2])
            idx1 = idx2
        elif(data_list[idx2] != 0 and data_list[idx1] == 0):
            encoded_list.append(idx2-idx1)
            idx1 = idx2
        idx2 += 1
            
    if(data_list[idx1] == 0 and idx2 == len(data_list)):
        encoded_list.append(0)
    elif(data_list[idx1] != 0 and idx2 == len(data_list)):
        encoded_list.append(idx1-idx2)
        encoded_list.extend(data_list[idx1:idx2])
        
    return encoded_list
            

def run_length_decoding(data_list, I):
    decoded_list = np.zeros((I*I), dtype=np.int16)
    decoded_idx = 0
    data_idx = 0
    
    if(data_list[0] != 0):
        while(decoded_idx < I*I):
            val = data_list[data_idx]
            if(data_list[data_idx] < 0):
                decoded_list[decoded_idx:decoded_idx+abs(val)] = data_list[data_idx+1:data_idx+abs(val)+1]
                decoded_idx += abs(val)
                data_idx += (abs(val) + 1)
            elif(data_list[data_idx] > 0):
                # already filled with zeros
                decoded_idx += abs(val)
                data_idx += 1
            else: # == 0
                data_idx += 1
                break
        return data_idx, decoded_list
    else:
        data_idx = 1
        return data_idx, decoded_list

def golomb_encoding(data_list):
    binary = Bits()
    for val in data_list:
        if(val <= 0):
            mapped_val = -2*val + 1
        else:
            mapped_val = 2*val
        val_binary = Bits(bin(mapped_val))
        if(val_binary.len > 1):
            val_binary = Bits('0b' + '0'*(val_binary.len-1)) + val_binary
        binary += val_binary
    return binary
                    
def golomb_decoding(binary):
    decoded_values = []
    zero_count = 0
    i = 0
    while( i < binary.len):
        if binary[i]:
            val = binary[i:i+zero_count+1].uint - 1
            if( val % 2 == 0):
                decoded_values.append(int(-val/2))
            else:
                decoded_values.append(int((val+1)/2))
            i += zero_count + 1
            zero_count = 0
        else:
            zero_count += 1
            i += 1
    return decoded_values
        
def add_with_saturation(block1, block2):
    return np.clip(block1 + block2, 0, 255).astype(np.uint8)

def get_bit_sizes(OUTPUT_DIR, FRAME_NUMS):
    bit_sizes = []
    for frame_num in range(FRAME_NUMS):
        encode_binary_path = f"{OUTPUT_DIR}/encoded_binary/binary_frame_{frame_num}"
        bit_sizes.append(os.path.getsize(encode_binary_path) * 8)
    return bit_sizes

def compare_decoded_frames(file_name, OUTPUT_DIR,FRAME_NUMS,HEIGHT, WIDTH):
    psnr_values = []
    for frame_num in range(FRAME_NUMS):
        original_filename = f"{file_name}_y_frames/y_frame_{frame_num}.y"
        decoded_filename = f"{OUTPUT_DIR}/y_frames_decoded/y_decoded_frame_{frame_num}.y"

        with open(original_filename, 'rb') as f:
            original_frame = np.frombuffer(f.read(), dtype=np.uint8).reshape(HEIGHT, WIDTH)

        with open(decoded_filename, 'rb') as f:
            decoded_frame = np.frombuffer(f.read(), dtype=np.uint8).reshape(HEIGHT, WIDTH)

        psnr_value = psnr(original_frame, decoded_frame)
        psnr_values.append(psnr_value)
    return psnr_values

def compare_reconstructed_frames(file_name, OUTPUT_DIR,FRAME_NUMS,HEIGHT, WIDTH):
    psnr_values = []
    for frame_num in range(FRAME_NUMS):
        original_filename = f"{file_name}_y_frames/y_frame_{frame_num}.y"
        decoded_filename = f"{OUTPUT_DIR}/y_frames_reconstructed/y_reconstructed_frame_{frame_num}.y"

        with open(original_filename, 'rb') as f:
            original_frame = np.frombuffer(f.read(), dtype=np.uint8).reshape(HEIGHT, WIDTH)

        with open(decoded_filename, 'rb') as f:
            decoded_frame = np.frombuffer(f.read(), dtype=np.uint8).reshape(HEIGHT, WIDTH)

        psnr_value = psnr(original_frame, decoded_frame)
        psnr_values.append(psnr_value)
    return psnr_values

def scene_change_threshold(width, height, QP):
    base_threshold = width * height  # Frame size factor
    QP_adjustment_factor = np.exp(5.58023370e-04*np.power(QP, 4))*np.exp(-9.38015025e-03*np.power(QP, 3))*np.exp(1.80101666e-02*np.power(QP, 2))*np.exp(-2.62071238e-01*np.power(QP, 1))*np.exp(1.75324455)
    tolerance_factor = 0.9
    return base_threshold * QP_adjustment_factor * tolerance_factor

def join_threads(thread_list):
    for thread in thread_list:
        thread.join()

if __name__ == "__main__":
    a = np.array([[-114,  -81,   59,  127,  123,  120,  107,  143],
                    [-137,  -84,   87,  137,  108,  146,  139,   98],
                    [ -96,  -98,   74,  124,   19,    2,   18,   10],
                    [-134,  -86,   91,  109,    9,   41,  110,  138],
                    [-121,  -72,   73,   88,   10,   -1,    9,   39],
                    [-104,  -88,   59,  142,  116,   61,   -4,   -2],
                    [-140,  -87,   85,  134,  102,  123,   78,    2],
                    [-112,  -99,   73,  103,   -1,   13,   27,  -21]])
    
    b = np.full((8,8), 128, dtype=np.int16)
    
    print(psnr(a, b))