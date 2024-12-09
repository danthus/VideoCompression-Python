import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def reset_parameters():
    frame_nums = 10
    I = 16
    R = 4
    QP = 1
    nRefFrames = 1
    VBSEnable = False
    const = 0.08
    const2 = 0.01
    FMEEnable = False
    FastME = False
    return frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME

def read_csv_by(directory, filename, read_method):
    data_set = []
    with open(f'{directory}/{filename}', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        skip_header = True
        for row in csv_reader:
            if skip_header: # Skip first
                skip_header = False
                continue
            data = read_method(row)
            data_set.append(data)
    return data_set

def read_k_col_as(row, k ,data_type):
    return data_type(row[k])

def read_time(row):
    return read_k_col_as(row, 1, float)

def read_RD(row):
    return [read_k_col_as(row, 0, float), read_k_col_as(row, 1, int)]

def read_VBS_percentage(row):
    return read_k_col_as(row, 1, float)

# Read all data
def read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME):
    directory = f'FRAME{frame_nums}I{I}R{R}QP{QP}-nref{nRefFrames}{VBSEnable}intra{const}inter{const2}{FMEEnable}{FastME}'
    encoding_times = read_csv_by(directory, "encoding_times.csv", read_time)
    decoding_times = read_csv_by(directory, "decoding_times.csv", read_time)
    RDs = read_csv_by(directory, "RD.csv", read_RD)
    VBS_percentages = read_csv_by(directory, "VBS_percentages.csv", read_VBS_percentage)
    return encoding_times, decoding_times, RDs, VBS_percentages

def save_curves_plot(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, OUTPUT_DIR, plot_title, x_name, y_name, curve_names):
    
    # Create plot
    plt.figure(figsize=(8, 6))

    # Plot curves
    for i, (x, y) in enumerate(zip([x1, x2, x3, x4, x5, x6], [y1, y2, y3, y4, y5, y6])):
        plt.plot(x, y, label=curve_names[i], linestyle='-', marker='o')
    
    # Init title and labels
    plt.legend()
    plt.title(plot_title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    
    # Create directory
    if not os.path.exists(f"{OUTPUT_DIR}"):
        os.makedirs(f"{OUTPUT_DIR}")
    # Save curves plot
    plt.savefig(os.path.join(OUTPUT_DIR, f"{plot_title}.png"))

    plt.clf()
    plt.close()


def plot_main():
    frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME = reset_parameters()
    # _QP1_nRef1_false_false_false
    QP = 1
    encoding_times_QP1_nRef1_false_false_false, decoding_times_QP1_nRef1_false_false_false, RDs_QP1_nRef1_false_false_false, VBS_percentages_QP1_nRef1_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP2_nRef1_false_false_false
    QP = 2
    encoding_times_QP2_nRef1_false_false_false, decoding_times_QP2_nRef1_false_false_false, RDs_QP2_nRef1_false_false_false, VBS_percentages_QP2_nRef1_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP3_nRef1_false_false_false
    QP = 3
    encoding_times_QP3_nRef1_false_false_false, decoding_times_QP3_nRef1_false_false_false, RDs_QP3_nRef1_false_false_false, VBS_percentages_QP3_nRef1_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP4_nRef1_false_false_false
    QP = 4
    encoding_times_QP4_nRef1_false_false_false, decoding_times_QP4_nRef1_false_false_false, RDs_QP4_nRef1_false_false_false, VBS_percentages_QP4_nRef1_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP5_nRef1_false_false_false
    QP = 5
    encoding_times_QP5_nRef1_false_false_false, decoding_times_QP5_nRef1_false_false_false, RDs_QP5_nRef1_false_false_false, VBS_percentages_QP5_nRef1_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP6_nRef1_false_false_false
    QP = 6
    encoding_times_QP6_nRef1_false_false_false, decoding_times_QP6_nRef1_false_false_false, RDs_QP6_nRef1_false_false_false, VBS_percentages_QP6_nRef1_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP7_nRef1_false_false_false
    QP = 7
    encoding_times_QP7_nRef1_false_false_false, decoding_times_QP7_nRef1_false_false_false, RDs_QP7_nRef1_false_false_false, VBS_percentages_QP7_nRef1_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP1_nRef1_false_false_false
    QP = 8
    encoding_times_QP8_nRef1_false_false_false, decoding_times_QP8_nRef1_false_false_false, RDs_QP8_nRef1_false_false_false, VBS_percentages_QP8_nRef1_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP9_nRef1_false_false_false
    QP = 9
    encoding_times_QP9_nRef1_false_false_false, decoding_times_QP9_nRef1_false_false_false, RDs_QP9_nRef1_false_false_false, VBS_percentages_QP9_nRef1_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP10_nRef1_false_false_false
    QP = 10
    encoding_times_QP10_nRef1_false_false_false, decoding_times_QP10_nRef1_false_false_false, RDs_QP10_nRef1_false_false_false, VBS_percentages_QP10_nRef1_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    
    
    frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME = reset_parameters()
    # nRef4
    nRefFrames = 4
    # _QP1_nRef4_false_false_false
    QP = 1
    encoding_times_QP1_nRef4_false_false_false, decoding_times_QP1_nRef4_false_false_false, RDs_QP1_nRef4_false_false_false, VBS_percentages_QP1_nRef4_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP2_nRef4_false_false_false
    QP = 2
    encoding_times_QP2_nRef4_false_false_false, decoding_times_QP2_nRef4_false_false_false, RDs_QP2_nRef4_false_false_false, VBS_percentages_QP2_nRef4_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP3_nRef4_false_false_false
    QP = 3
    encoding_times_QP3_nRef4_false_false_false, decoding_times_QP3_nRef4_false_false_false, RDs_QP3_nRef4_false_false_false, VBS_percentages_QP3_nRef4_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP4_nRef4_false_false_false
    QP = 4
    encoding_times_QP4_nRef4_false_false_false, decoding_times_QP4_nRef4_false_false_false, RDs_QP4_nRef4_false_false_false, VBS_percentages_QP4_nRef4_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP5_nRef4_false_false_false
    QP = 5
    encoding_times_QP5_nRef4_false_false_false, decoding_times_QP5_nRef4_false_false_false, RDs_QP5_nRef4_false_false_false, VBS_percentages_QP5_nRef4_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP6_nRef4_false_false_false
    QP = 6
    encoding_times_QP6_nRef4_false_false_false, decoding_times_QP6_nRef4_false_false_false, RDs_QP6_nRef4_false_false_false, VBS_percentages_QP6_nRef4_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP7_nRef4_false_false_false
    QP = 7
    encoding_times_QP7_nRef4_false_false_false, decoding_times_QP7_nRef4_false_false_false, RDs_QP7_nRef4_false_false_false, VBS_percentages_QP7_nRef4_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP1_nRef4_false_false_false
    QP = 8
    encoding_times_QP8_nRef4_false_false_false, decoding_times_QP8_nRef4_false_false_false, RDs_QP8_nRef4_false_false_false, VBS_percentages_QP8_nRef4_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP9_nRef4_false_false_false
    QP = 9
    encoding_times_QP9_nRef4_false_false_false, decoding_times_QP9_nRef4_false_false_false, RDs_QP9_nRef4_false_false_false, VBS_percentages_QP9_nRef4_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP10_nRef4_false_false_false
    QP = 10
    encoding_times_QP10_nRef4_false_false_false, decoding_times_QP10_nRef4_false_false_false, RDs_QP10_nRef4_false_false_false, VBS_percentages_QP10_nRef4_false_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)

    frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME = reset_parameters()
    # VBS True
    VBSEnable = True
    # _QP1_nRef1_true_false_false
    QP = 1
    encoding_times_QP1_nRef1_true_false_false, decoding_times_QP1_nRef1_true_false_false, RDs_QP1_nRef1_true_false_false, VBS_percentages_QP1_nRef1_true_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP2_nRef1_true_false_false
    QP = 2
    encoding_times_QP2_nRef1_true_false_false, decoding_times_QP2_nRef1_true_false_false, RDs_QP2_nRef1_true_false_false, VBS_percentages_QP2_nRef1_true_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP3_nRef1_true_false_false
    QP = 3
    encoding_times_QP3_nRef1_true_false_false, decoding_times_QP3_nRef1_true_false_false, RDs_QP3_nRef1_true_false_false, VBS_percentages_QP3_nRef1_true_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP4_nRef1_true_false_false
    QP = 4
    encoding_times_QP4_nRef1_true_false_false, decoding_times_QP4_nRef1_true_false_false, RDs_QP4_nRef1_true_false_false, VBS_percentages_QP4_nRef1_true_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP5_nRef1_true_false_false
    QP = 5
    encoding_times_QP5_nRef1_true_false_false, decoding_times_QP5_nRef1_true_false_false, RDs_QP5_nRef1_true_false_false, VBS_percentages_QP5_nRef1_true_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP6_nRef1_true_false_false
    QP = 6
    encoding_times_QP6_nRef1_true_false_false, decoding_times_QP6_nRef1_true_false_false, RDs_QP6_nRef1_true_false_false, VBS_percentages_QP6_nRef1_true_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP7_nRef1_true_false_false
    QP = 7
    encoding_times_QP7_nRef1_true_false_false, decoding_times_QP7_nRef1_true_false_false, RDs_QP7_nRef1_true_false_false, VBS_percentages_QP7_nRef1_true_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP1_nRef1_true_false_false
    QP = 8
    encoding_times_QP8_nRef1_true_false_false, decoding_times_QP8_nRef1_true_false_false, RDs_QP8_nRef1_true_false_false, VBS_percentages_QP8_nRef1_true_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP9_nRef1_true_false_false
    QP = 9
    encoding_times_QP9_nRef1_true_false_false, decoding_times_QP9_nRef1_true_false_false, RDs_QP9_nRef1_true_false_false, VBS_percentages_QP9_nRef1_true_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP10_nRef1_true_false_false
    QP = 10
    encoding_times_QP10_nRef1_true_false_false, decoding_times_QP10_nRef1_true_false_false, RDs_QP10_nRef1_true_false_false, VBS_percentages_QP10_nRef1_true_false_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)

    frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME = reset_parameters()
    # FME true
    FMEEnable = True
    # _QP1_nRef1_false_true_false
    QP = 1
    encoding_times_QP1_nRef1_false_true_false, decoding_times_QP1_nRef1_false_true_false, RDs_QP1_nRef1_false_true_false, VBS_percentages_QP1_nRef1_false_true_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP2_nRef1_false_true_false
    QP = 2
    encoding_times_QP2_nRef1_false_true_false, decoding_times_QP2_nRef1_false_true_false, RDs_QP2_nRef1_false_true_false, VBS_percentages_QP2_nRef1_false_true_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP3_nRef1_false_true_false
    QP = 3
    encoding_times_QP3_nRef1_false_true_false, decoding_times_QP3_nRef1_false_true_false, RDs_QP3_nRef1_false_true_false, VBS_percentages_QP3_nRef1_false_true_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP4_nRef1_false_true_false
    QP = 4
    encoding_times_QP4_nRef1_false_true_false, decoding_times_QP4_nRef1_false_true_false, RDs_QP4_nRef1_false_true_false, VBS_percentages_QP4_nRef1_false_true_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP5_nRef1_false_true_false
    QP = 5
    encoding_times_QP5_nRef1_false_true_false, decoding_times_QP5_nRef1_false_true_false, RDs_QP5_nRef1_false_true_false, VBS_percentages_QP5_nRef1_false_true_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP6_nRef1_false_true_false
    QP = 6
    encoding_times_QP6_nRef1_false_true_false, decoding_times_QP6_nRef1_false_true_false, RDs_QP6_nRef1_false_true_false, VBS_percentages_QP6_nRef1_false_true_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP7_nRef1_false_true_false
    QP = 7
    encoding_times_QP7_nRef1_false_true_false, decoding_times_QP7_nRef1_false_true_false, RDs_QP7_nRef1_false_true_false, VBS_percentages_QP7_nRef1_false_true_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP1_nRef1_false_true_false
    QP = 8
    encoding_times_QP8_nRef1_false_true_false, decoding_times_QP8_nRef1_false_true_false, RDs_QP8_nRef1_false_true_false, VBS_percentages_QP8_nRef1_false_true_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP9_nRef1_false_true_false
    QP = 9
    encoding_times_QP9_nRef1_false_true_false, decoding_times_QP9_nRef1_false_true_false, RDs_QP9_nRef1_false_true_false, VBS_percentages_QP9_nRef1_false_true_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP10_nRef1_false_true_false
    QP = 10
    encoding_times_QP10_nRef1_false_true_false, decoding_times_QP10_nRef1_false_true_false, RDs_QP10_nRef1_false_true_false, VBS_percentages_QP10_nRef1_false_true_false = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)

    frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME = reset_parameters()
    # FastME true
    FastME = True
    # _QP1_nRef1_false_false_true
    QP = 1
    encoding_times_QP1_nRef1_false_false_true, decoding_times_QP1_nRef1_false_false_true, RDs_QP1_nRef1_false_false_true, VBS_percentages_QP1_nRef1_false_false_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP2_nRef1_false_false_true
    QP = 2
    encoding_times_QP2_nRef1_false_false_true, decoding_times_QP2_nRef1_false_false_true, RDs_QP2_nRef1_false_false_true, VBS_percentages_QP2_nRef1_false_false_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP3_nRef1_false_false_true
    QP = 3
    encoding_times_QP3_nRef1_false_false_true, decoding_times_QP3_nRef1_false_false_true, RDs_QP3_nRef1_false_false_true, VBS_percentages_QP3_nRef1_false_false_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP4_nRef1_false_false_true
    QP = 4
    encoding_times_QP4_nRef1_false_false_true, decoding_times_QP4_nRef1_false_false_true, RDs_QP4_nRef1_false_false_true, VBS_percentages_QP4_nRef1_false_false_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP5_nRef1_false_false_true
    QP = 5
    encoding_times_QP5_nRef1_false_false_true, decoding_times_QP5_nRef1_false_false_true, RDs_QP5_nRef1_false_false_true, VBS_percentages_QP5_nRef1_false_false_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP6_nRef1_false_false_true
    QP = 6
    encoding_times_QP6_nRef1_false_false_true, decoding_times_QP6_nRef1_false_false_true, RDs_QP6_nRef1_false_false_true, VBS_percentages_QP6_nRef1_false_false_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP7_nRef1_false_false_true
    QP = 7
    encoding_times_QP7_nRef1_false_false_true, decoding_times_QP7_nRef1_false_false_true, RDs_QP7_nRef1_false_false_true, VBS_percentages_QP7_nRef1_false_false_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP1_nRef1_false_false_true
    QP = 8
    encoding_times_QP8_nRef1_false_false_true, decoding_times_QP8_nRef1_false_false_true, RDs_QP8_nRef1_false_false_true, VBS_percentages_QP8_nRef1_false_false_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP9_nRef1_false_false_true
    QP = 9
    encoding_times_QP9_nRef1_false_false_true, decoding_times_QP9_nRef1_false_false_true, RDs_QP9_nRef1_false_false_true, VBS_percentages_QP9_nRef1_false_false_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP10_nRef1_false_false_true
    QP = 10
    encoding_times_QP10_nRef1_false_false_true, decoding_times_QP10_nRef1_false_false_true, RDs_QP10_nRef1_false_false_true, VBS_percentages_QP10_nRef1_false_false_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    
    frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME = reset_parameters()
    # nRef4, VBS true, FME true, FastME true
    nRefFrames = 4
    VBSEnable = True
    FMEEnable = True
    FastME = True
    # _QP1_nRef4_true_true_true
    QP = 1
    encoding_times_QP1_nRef4_true_true_true, decoding_times_QP1_nRef4_true_true_true, RDs_QP1_nRef4_true_true_true, VBS_percentages_QP1_nRef4_true_true_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP2_nRef4_true_true_true
    QP = 2
    encoding_times_QP2_nRef4_true_true_true, decoding_times_QP2_nRef4_true_true_true, RDs_QP2_nRef4_true_true_true, VBS_percentages_QP2_nRef4_true_true_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP3_nRef4_true_true_true
    QP = 3
    encoding_times_QP3_nRef4_true_true_true, decoding_times_QP3_nRef4_true_true_true, RDs_QP3_nRef4_true_true_true, VBS_percentages_QP3_nRef4_true_true_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP4_nRef4_true_true_true
    QP = 4
    encoding_times_QP4_nRef4_true_true_true, decoding_times_QP4_nRef4_true_true_true, RDs_QP4_nRef4_true_true_true, VBS_percentages_QP4_nRef4_true_true_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP5_nRef4_true_true_true
    QP = 5
    encoding_times_QP5_nRef4_true_true_true, decoding_times_QP5_nRef4_true_true_true, RDs_QP5_nRef4_true_true_true, VBS_percentages_QP5_nRef4_true_true_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP6_nRef4_true_true_true
    QP = 6
    encoding_times_QP6_nRef4_true_true_true, decoding_times_QP6_nRef4_true_true_true, RDs_QP6_nRef4_true_true_true, VBS_percentages_QP6_nRef4_true_true_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP7_nRef4_true_true_true
    QP = 7
    encoding_times_QP7_nRef4_true_true_true, decoding_times_QP7_nRef4_true_true_true, RDs_QP7_nRef4_true_true_true, VBS_percentages_QP7_nRef4_true_true_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP1_nRef4_true_true_true
    QP = 8
    encoding_times_QP8_nRef4_true_true_true, decoding_times_QP8_nRef4_true_true_true, RDs_QP8_nRef4_true_true_true, VBS_percentages_QP8_nRef4_true_true_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP9_nRef4_true_true_true
    QP = 9
    encoding_times_QP9_nRef4_true_true_true, decoding_times_QP9_nRef4_true_true_true, RDs_QP9_nRef4_true_true_true, VBS_percentages_QP9_nRef4_true_true_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)
    # _QP10_nRef4_true_true_true
    QP = 10
    encoding_times_QP10_nRef4_true_true_true, decoding_times_QP10_nRef4_true_true_true, RDs_QP10_nRef4_true_true_true, VBS_percentages_QP10_nRef4_true_true_true = read_data(frame_nums, I, R, QP, nRefFrames, VBSEnable, const, const2, FMEEnable, FastME)

    curve_names = ["No feature", f'Only nRefFrames 4', "Only VBS", "Only FME", "Only FastME", "All features"]

    # Plot RD curves plots
    for frame_num in range(frame_nums):
        # _nRef1_false_false_false [x1, x2, x3, x4, ...] [y1, y2, y3, y4, ...]
        bit_size_nRef1_false_false_false = [RDs_QP1_nRef1_false_false_false[frame_num][1],
                                            RDs_QP2_nRef1_false_false_false[frame_num][1],
                                            RDs_QP3_nRef1_false_false_false[frame_num][1],
                                            RDs_QP4_nRef1_false_false_false[frame_num][1],
                                            RDs_QP5_nRef1_false_false_false[frame_num][1],
                                            RDs_QP6_nRef1_false_false_false[frame_num][1],
                                            RDs_QP7_nRef1_false_false_false[frame_num][1],
                                            RDs_QP8_nRef1_false_false_false[frame_num][1],
                                            RDs_QP9_nRef1_false_false_false[frame_num][1],
                                            RDs_QP10_nRef1_false_false_false[frame_num][1]]
        psnr_nRef1_false_false_false = [RDs_QP1_nRef1_false_false_false[frame_num][0],
                                        RDs_QP2_nRef1_false_false_false[frame_num][0],
                                        RDs_QP3_nRef1_false_false_false[frame_num][0],
                                        RDs_QP4_nRef1_false_false_false[frame_num][0],
                                        RDs_QP5_nRef1_false_false_false[frame_num][0],
                                        RDs_QP6_nRef1_false_false_false[frame_num][0],
                                        RDs_QP7_nRef1_false_false_false[frame_num][0],
                                        RDs_QP8_nRef1_false_false_false[frame_num][0],
                                        RDs_QP9_nRef1_false_false_false[frame_num][0],
                                        RDs_QP10_nRef1_false_false_false[frame_num][0]]
        
        # _nRef4_false_false_false [x1, x2, x3, x4, ...] [y1, y2, y3, y4, ...]
        bit_size_nRef4_false_false_false = [RDs_QP1_nRef4_false_false_false[frame_num][1],
                                            RDs_QP2_nRef4_false_false_false[frame_num][1],
                                            RDs_QP3_nRef4_false_false_false[frame_num][1],
                                            RDs_QP4_nRef4_false_false_false[frame_num][1],
                                            RDs_QP5_nRef4_false_false_false[frame_num][1],
                                            RDs_QP6_nRef4_false_false_false[frame_num][1],
                                            RDs_QP7_nRef4_false_false_false[frame_num][1],
                                            RDs_QP8_nRef4_false_false_false[frame_num][1],
                                            RDs_QP9_nRef4_false_false_false[frame_num][1],
                                            RDs_QP10_nRef4_false_false_false[frame_num][1]]
        psnr_nRef4_false_false_false = [RDs_QP1_nRef4_false_false_false[frame_num][0],
                                        RDs_QP2_nRef4_false_false_false[frame_num][0],
                                        RDs_QP3_nRef4_false_false_false[frame_num][0],
                                        RDs_QP4_nRef4_false_false_false[frame_num][0],
                                        RDs_QP5_nRef4_false_false_false[frame_num][0],
                                        RDs_QP6_nRef4_false_false_false[frame_num][0],
                                        RDs_QP7_nRef4_false_false_false[frame_num][0],
                                        RDs_QP8_nRef4_false_false_false[frame_num][0],
                                        RDs_QP9_nRef4_false_false_false[frame_num][0],
                                        RDs_QP10_nRef4_false_false_false[frame_num][0]]

        # _nRef1_true_false_false [x1, x2, x3, x4, ...] [y1, y2, y3, y4, ...]
        bit_size_nRef1_true_false_false = [RDs_QP1_nRef1_true_false_false[frame_num][1],
                                            RDs_QP2_nRef1_true_false_false[frame_num][1],
                                            RDs_QP3_nRef1_true_false_false[frame_num][1],
                                            RDs_QP4_nRef1_true_false_false[frame_num][1],
                                            RDs_QP5_nRef1_true_false_false[frame_num][1],
                                            RDs_QP6_nRef1_true_false_false[frame_num][1],
                                            RDs_QP7_nRef1_true_false_false[frame_num][1],
                                            RDs_QP8_nRef1_true_false_false[frame_num][1],
                                            RDs_QP9_nRef1_true_false_false[frame_num][1],
                                            RDs_QP10_nRef1_true_false_false[frame_num][1]]
        psnr_nRef1_true_false_false = [RDs_QP1_nRef1_true_false_false[frame_num][0],
                                        RDs_QP2_nRef1_true_false_false[frame_num][0],
                                        RDs_QP3_nRef1_true_false_false[frame_num][0],
                                        RDs_QP4_nRef1_true_false_false[frame_num][0],
                                        RDs_QP5_nRef1_true_false_false[frame_num][0],
                                        RDs_QP6_nRef1_true_false_false[frame_num][0],
                                        RDs_QP7_nRef1_true_false_false[frame_num][0],
                                        RDs_QP8_nRef1_true_false_false[frame_num][0],
                                        RDs_QP9_nRef1_true_false_false[frame_num][0],
                                        RDs_QP10_nRef1_true_false_false[frame_num][0]]

        # _nRef1_false_true_false [x1, x2, x3, x4, ...] [y1, y2, y3, y4, ...]
        bit_size_nRef1_false_true_false = [RDs_QP1_nRef1_false_true_false[frame_num][1],
                                            RDs_QP2_nRef1_false_true_false[frame_num][1],
                                            RDs_QP3_nRef1_false_true_false[frame_num][1],
                                            RDs_QP4_nRef1_false_true_false[frame_num][1],
                                            RDs_QP5_nRef1_false_true_false[frame_num][1],
                                            RDs_QP6_nRef1_false_true_false[frame_num][1],
                                            RDs_QP7_nRef1_false_true_false[frame_num][1],
                                            RDs_QP8_nRef1_false_true_false[frame_num][1],
                                            RDs_QP9_nRef1_false_true_false[frame_num][1],
                                            RDs_QP10_nRef1_false_true_false[frame_num][1]]
        psnr_nRef1_false_true_false = [RDs_QP1_nRef1_false_true_false[frame_num][0],
                                        RDs_QP2_nRef1_false_true_false[frame_num][0],
                                        RDs_QP3_nRef1_false_true_false[frame_num][0],
                                        RDs_QP4_nRef1_false_true_false[frame_num][0],
                                        RDs_QP5_nRef1_false_true_false[frame_num][0],
                                        RDs_QP6_nRef1_false_true_false[frame_num][0],
                                        RDs_QP7_nRef1_false_true_false[frame_num][0],
                                        RDs_QP8_nRef1_false_true_false[frame_num][0],
                                        RDs_QP9_nRef1_false_true_false[frame_num][0],
                                        RDs_QP10_nRef1_false_true_false[frame_num][0]]
        
        # _nRef4_false_false_true [x1, x2, x3, x4, ...] [y1, y2, y3, y4, ...]
        bit_size_nRef1_false_false_true = [RDs_QP1_nRef1_false_false_true[frame_num][1],
                                            RDs_QP2_nRef1_false_false_true[frame_num][1],
                                            RDs_QP3_nRef1_false_false_true[frame_num][1],
                                            RDs_QP4_nRef1_false_false_true[frame_num][1],
                                            RDs_QP5_nRef1_false_false_true[frame_num][1],
                                            RDs_QP6_nRef1_false_false_true[frame_num][1],
                                            RDs_QP7_nRef1_false_false_true[frame_num][1],
                                            RDs_QP8_nRef1_false_false_true[frame_num][1],
                                            RDs_QP9_nRef1_false_false_true[frame_num][1],
                                            RDs_QP10_nRef1_false_false_true[frame_num][1]]
        psnr_nRef1_false_false_true = [RDs_QP1_nRef1_false_false_true[frame_num][0],
                                        RDs_QP2_nRef1_false_false_true[frame_num][0],
                                        RDs_QP3_nRef1_false_false_true[frame_num][0],
                                        RDs_QP4_nRef1_false_false_true[frame_num][0],
                                        RDs_QP5_nRef1_false_false_true[frame_num][0],
                                        RDs_QP6_nRef1_false_false_true[frame_num][0],
                                        RDs_QP7_nRef1_false_false_true[frame_num][0],
                                        RDs_QP8_nRef1_false_false_true[frame_num][0],
                                        RDs_QP9_nRef1_false_false_true[frame_num][0],
                                        RDs_QP10_nRef1_false_false_true[frame_num][0]]

        # _nRef4_true_true_true [x1, x2, x3, x4, ...] [y1, y2, y3, y4, ...]
        bit_size_nRef4_true_true_true = [RDs_QP1_nRef4_true_true_true[frame_num][1],
                                            RDs_QP2_nRef4_true_true_true[frame_num][1],
                                            RDs_QP3_nRef4_true_true_true[frame_num][1],
                                            RDs_QP4_nRef4_true_true_true[frame_num][1],
                                            RDs_QP5_nRef4_true_true_true[frame_num][1],
                                            RDs_QP6_nRef4_true_true_true[frame_num][1],
                                            RDs_QP7_nRef4_true_true_true[frame_num][1],
                                            RDs_QP8_nRef4_true_true_true[frame_num][1],
                                            RDs_QP9_nRef4_true_true_true[frame_num][1],
                                            RDs_QP10_nRef4_true_true_true[frame_num][1]]
        psnr_nRef4_true_true_true = [RDs_QP1_nRef4_true_true_true[frame_num][0],
                                        RDs_QP2_nRef4_true_true_true[frame_num][0],
                                        RDs_QP3_nRef4_true_true_true[frame_num][0],
                                        RDs_QP4_nRef4_true_true_true[frame_num][0],
                                        RDs_QP5_nRef4_true_true_true[frame_num][0],
                                        RDs_QP6_nRef4_true_true_true[frame_num][0],
                                        RDs_QP7_nRef4_true_true_true[frame_num][0],
                                        RDs_QP8_nRef4_true_true_true[frame_num][0],
                                        RDs_QP9_nRef4_true_true_true[frame_num][0],
                                        RDs_QP10_nRef4_true_true_true[frame_num][0]]

        save_curves_plot(bit_size_nRef1_false_false_false, psnr_nRef1_false_false_false, bit_size_nRef4_false_false_false, psnr_nRef4_false_false_false,
                         bit_size_nRef1_true_false_false, psnr_nRef1_true_false_false, bit_size_nRef1_false_true_false, psnr_nRef1_false_true_false,
                         bit_size_nRef1_false_false_true, psnr_nRef1_false_false_true, bit_size_nRef4_true_true_true, psnr_nRef4_true_true_true,
                         "RD_plots", f'RD_plot_Frame_{frame_num}', "bit size (in bits)", "psnr", curve_names)
    # Plot average RD curves plot
    # _nRef1_false_false_false_avg [x1, x2, x3, x4, ...] [y1, y2, y3, y4, ...]
    bit_size_nRef1_false_false_false_avg = [int(np.mean(RDs_QP1_nRef1_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP2_nRef1_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP3_nRef1_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP4_nRef1_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP5_nRef1_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP6_nRef1_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP7_nRef1_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP8_nRef1_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP9_nRef1_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP10_nRef1_false_false_false, axis=0)[1])]
    psnr_nRef1_false_false_false_avg = [int(np.mean(RDs_QP1_nRef1_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP2_nRef1_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP3_nRef1_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP4_nRef1_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP5_nRef1_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP6_nRef1_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP7_nRef1_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP8_nRef1_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP9_nRef1_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP10_nRef1_false_false_false, axis=0)[0])]
    
    # _nRef4_false_false_false_avg [x1, x2, x3, x4, ...] [y1, y2, y3, y4, ...]
    bit_size_nRef4_false_false_false_avg = [int(np.mean(RDs_QP1_nRef4_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP2_nRef4_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP3_nRef4_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP4_nRef4_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP5_nRef4_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP6_nRef4_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP7_nRef4_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP8_nRef4_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP9_nRef4_false_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP10_nRef4_false_false_false, axis=0)[1])]
    psnr_nRef4_false_false_false_avg = [int(np.mean(RDs_QP1_nRef4_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP2_nRef4_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP3_nRef4_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP4_nRef4_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP5_nRef4_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP6_nRef4_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP7_nRef4_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP8_nRef4_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP9_nRef4_false_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP10_nRef4_false_false_false, axis=0)[0])]
    
    # _nRef1_true_false_false_avg [x1, x2, x3, x4, ...] [y1, y2, y3, y4, ...]
    bit_size_nRef1_true_false_false_avg = [int(np.mean(RDs_QP1_nRef1_true_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP2_nRef1_true_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP3_nRef1_true_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP4_nRef1_true_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP5_nRef1_true_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP6_nRef1_true_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP7_nRef1_true_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP8_nRef1_true_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP9_nRef1_true_false_false, axis=0)[1]),
                                            int(np.mean(RDs_QP10_nRef1_true_false_false, axis=0)[1])]
    psnr_nRef1_true_false_false_avg = [int(np.mean(RDs_QP1_nRef1_true_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP2_nRef1_true_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP3_nRef1_true_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP4_nRef1_true_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP5_nRef1_true_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP6_nRef1_true_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP7_nRef1_true_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP8_nRef1_true_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP9_nRef1_true_false_false, axis=0)[0]),
                                        int(np.mean(RDs_QP10_nRef1_true_false_false, axis=0)[0])]

    # _nRef1_false_true_false_avg [x1, x2, x3, x4, ...] [y1, y2, y3, y4, ...]
    bit_size_nRef1_false_true_false_avg = [int(np.mean(RDs_QP1_nRef1_false_true_false, axis=0)[1]),
                                            int(np.mean(RDs_QP2_nRef1_false_true_false, axis=0)[1]),
                                            int(np.mean(RDs_QP3_nRef1_false_true_false, axis=0)[1]),
                                            int(np.mean(RDs_QP4_nRef1_false_true_false, axis=0)[1]),
                                            int(np.mean(RDs_QP5_nRef1_false_true_false, axis=0)[1]),
                                            int(np.mean(RDs_QP6_nRef1_false_true_false, axis=0)[1]),
                                            int(np.mean(RDs_QP7_nRef1_false_true_false, axis=0)[1]),
                                            int(np.mean(RDs_QP8_nRef1_false_true_false, axis=0)[1]),
                                            int(np.mean(RDs_QP9_nRef1_false_true_false, axis=0)[1]),
                                            int(np.mean(RDs_QP10_nRef1_false_true_false, axis=0)[1])]
    psnr_nRef1_false_true_false_avg = [int(np.mean(RDs_QP1_nRef1_false_true_false, axis=0)[0]),
                                        int(np.mean(RDs_QP2_nRef1_false_true_false, axis=0)[0]),
                                        int(np.mean(RDs_QP3_nRef1_false_true_false, axis=0)[0]),
                                        int(np.mean(RDs_QP4_nRef1_false_true_false, axis=0)[0]),
                                        int(np.mean(RDs_QP5_nRef1_false_true_false, axis=0)[0]),
                                        int(np.mean(RDs_QP6_nRef1_false_true_false, axis=0)[0]),
                                        int(np.mean(RDs_QP7_nRef1_false_true_false, axis=0)[0]),
                                        int(np.mean(RDs_QP8_nRef1_false_true_false, axis=0)[0]),
                                        int(np.mean(RDs_QP9_nRef1_false_true_false, axis=0)[0]),
                                        int(np.mean(RDs_QP10_nRef1_false_true_false, axis=0)[0])]

    # _nRef1_false_false_true_avg [x1, x2, x3, x4, ...] [y1, y2, y3, y4, ...]
    bit_size_nRef1_false_false_true_avg = [int(np.mean(RDs_QP1_nRef1_false_false_true, axis=0)[1]),
                                            int(np.mean(RDs_QP2_nRef1_false_false_true, axis=0)[1]),
                                            int(np.mean(RDs_QP3_nRef1_false_false_true, axis=0)[1]),
                                            int(np.mean(RDs_QP4_nRef1_false_false_true, axis=0)[1]),
                                            int(np.mean(RDs_QP5_nRef1_false_false_true, axis=0)[1]),
                                            int(np.mean(RDs_QP6_nRef1_false_false_true, axis=0)[1]),
                                            int(np.mean(RDs_QP7_nRef1_false_false_true, axis=0)[1]),
                                            int(np.mean(RDs_QP8_nRef1_false_false_true, axis=0)[1]),
                                            int(np.mean(RDs_QP9_nRef1_false_false_true, axis=0)[1]),
                                            int(np.mean(RDs_QP10_nRef1_false_false_true, axis=0)[1])]
    psnr_nRef1_false_false_true_avg = [int(np.mean(RDs_QP1_nRef1_false_false_true, axis=0)[0]),
                                        int(np.mean(RDs_QP2_nRef1_false_false_true, axis=0)[0]),
                                        int(np.mean(RDs_QP3_nRef1_false_false_true, axis=0)[0]),
                                        int(np.mean(RDs_QP4_nRef1_false_false_true, axis=0)[0]),
                                        int(np.mean(RDs_QP5_nRef1_false_false_true, axis=0)[0]),
                                        int(np.mean(RDs_QP6_nRef1_false_false_true, axis=0)[0]),
                                        int(np.mean(RDs_QP7_nRef1_false_false_true, axis=0)[0]),
                                        int(np.mean(RDs_QP8_nRef1_false_false_true, axis=0)[0]),
                                        int(np.mean(RDs_QP9_nRef1_false_false_true, axis=0)[0]),
                                        int(np.mean(RDs_QP10_nRef1_false_false_true, axis=0)[0])]

    # _nRef4_true_true_true_avg [x1, x2, x3, x4, ...] [y1, y2, y3, y4, ...]
    bit_size_nRef4_true_true_true_avg = [int(np.mean(RDs_QP1_nRef4_true_true_true, axis=0)[1]),
                                            int(np.mean(RDs_QP2_nRef4_true_true_true, axis=0)[1]),
                                            int(np.mean(RDs_QP3_nRef4_true_true_true, axis=0)[1]),
                                            int(np.mean(RDs_QP4_nRef4_true_true_true, axis=0)[1]),
                                            int(np.mean(RDs_QP5_nRef4_true_true_true, axis=0)[1]),
                                            int(np.mean(RDs_QP6_nRef4_true_true_true, axis=0)[1]),
                                            int(np.mean(RDs_QP7_nRef4_true_true_true, axis=0)[1]),
                                            int(np.mean(RDs_QP8_nRef4_true_true_true, axis=0)[1]),
                                            int(np.mean(RDs_QP9_nRef4_true_true_true, axis=0)[1]),
                                            int(np.mean(RDs_QP10_nRef4_true_true_true, axis=0)[1])]
    psnr_nRef4_true_true_true_avg = [int(np.mean(RDs_QP1_nRef4_true_true_true, axis=0)[0]),
                                        int(np.mean(RDs_QP2_nRef4_true_true_true, axis=0)[0]),
                                        int(np.mean(RDs_QP3_nRef4_true_true_true, axis=0)[0]),
                                        int(np.mean(RDs_QP4_nRef4_true_true_true, axis=0)[0]),
                                        int(np.mean(RDs_QP5_nRef4_true_true_true, axis=0)[0]),
                                        int(np.mean(RDs_QP6_nRef4_true_true_true, axis=0)[0]),
                                        int(np.mean(RDs_QP7_nRef4_true_true_true, axis=0)[0]),
                                        int(np.mean(RDs_QP8_nRef4_true_true_true, axis=0)[0]),
                                        int(np.mean(RDs_QP9_nRef4_true_true_true, axis=0)[0]),
                                        int(np.mean(RDs_QP10_nRef4_true_true_true, axis=0)[0])]

    save_curves_plot(bit_size_nRef1_false_false_false_avg, psnr_nRef1_false_false_false_avg, bit_size_nRef4_false_false_false_avg, psnr_nRef4_false_false_false_avg,
                     bit_size_nRef1_true_false_false_avg, psnr_nRef1_true_false_false_avg, bit_size_nRef1_false_true_false_avg, psnr_nRef1_false_true_false_avg,
                     bit_size_nRef1_false_false_true_avg, psnr_nRef1_false_false_true_avg, bit_size_nRef4_true_true_true_avg, psnr_nRef4_true_true_true_avg,
                     "RD_plots", "Average RD plot", "bit size (in bits)", "psnr", curve_names)
    

    # Plot Encoding time VS QP plots
    QPs = [1,2,3,4,5,6,7,8,9,10]
    for frame_num in range(frame_nums):
        # _nRef1_false_false_false [t1, t2, t3, t4, ...]
        encoding_times_nRef1_false_false_false = [encoding_times_QP1_nRef1_false_false_false[frame_num],
                                                  encoding_times_QP2_nRef1_false_false_false[frame_num],
                                                  encoding_times_QP3_nRef1_false_false_false[frame_num],
                                                  encoding_times_QP4_nRef1_false_false_false[frame_num],
                                                  encoding_times_QP5_nRef1_false_false_false[frame_num],
                                                  encoding_times_QP6_nRef1_false_false_false[frame_num],
                                                  encoding_times_QP7_nRef1_false_false_false[frame_num],
                                                  encoding_times_QP8_nRef1_false_false_false[frame_num],
                                                  encoding_times_QP9_nRef1_false_false_false[frame_num],
                                                  encoding_times_QP10_nRef1_false_false_false[frame_num]]
        
        # _nRef4_false_false_false [t1, t2, t3, t4, ...]
        encoding_times_nRef4_false_false_false = [encoding_times_QP1_nRef4_false_false_false[frame_num],
                                                  encoding_times_QP2_nRef4_false_false_false[frame_num],
                                                  encoding_times_QP3_nRef4_false_false_false[frame_num],
                                                  encoding_times_QP4_nRef4_false_false_false[frame_num],
                                                  encoding_times_QP5_nRef4_false_false_false[frame_num],
                                                  encoding_times_QP6_nRef4_false_false_false[frame_num],
                                                  encoding_times_QP7_nRef4_false_false_false[frame_num],
                                                  encoding_times_QP8_nRef4_false_false_false[frame_num],
                                                  encoding_times_QP9_nRef4_false_false_false[frame_num],
                                                  encoding_times_QP10_nRef4_false_false_false[frame_num]]
        
        # _nRef1_true_false_false [t1, t2, t3, t4, ...]
        encoding_times_nRef1_true_false_false = [encoding_times_QP1_nRef1_true_false_false[frame_num],
                                                  encoding_times_QP2_nRef1_true_false_false[frame_num],
                                                  encoding_times_QP3_nRef1_true_false_false[frame_num],
                                                  encoding_times_QP4_nRef1_true_false_false[frame_num],
                                                  encoding_times_QP5_nRef1_true_false_false[frame_num],
                                                  encoding_times_QP6_nRef1_true_false_false[frame_num],
                                                  encoding_times_QP7_nRef1_true_false_false[frame_num],
                                                  encoding_times_QP8_nRef1_true_false_false[frame_num],
                                                  encoding_times_QP9_nRef1_true_false_false[frame_num],
                                                  encoding_times_QP10_nRef1_true_false_false[frame_num]]
        
        # _nRef1_false_true_false [t1, t2, t3, t4, ...]
        encoding_times_nRef1_false_true_false = [encoding_times_QP1_nRef1_false_true_false[frame_num],
                                                  encoding_times_QP2_nRef1_false_true_false[frame_num],
                                                  encoding_times_QP3_nRef1_false_true_false[frame_num],
                                                  encoding_times_QP4_nRef1_false_true_false[frame_num],
                                                  encoding_times_QP5_nRef1_false_true_false[frame_num],
                                                  encoding_times_QP6_nRef1_false_true_false[frame_num],
                                                  encoding_times_QP7_nRef1_false_true_false[frame_num],
                                                  encoding_times_QP8_nRef1_false_true_false[frame_num],
                                                  encoding_times_QP9_nRef1_false_true_false[frame_num],
                                                  encoding_times_QP10_nRef1_false_true_false[frame_num]]
        
        # _nRef1_false_false_true [t1, t2, t3, t4, ...]
        encoding_times_nRef1_false_false_true = [encoding_times_QP1_nRef1_false_false_true[frame_num],
                                                  encoding_times_QP2_nRef1_false_false_true[frame_num],
                                                  encoding_times_QP3_nRef1_false_false_true[frame_num],
                                                  encoding_times_QP4_nRef1_false_false_true[frame_num],
                                                  encoding_times_QP5_nRef1_false_false_true[frame_num],
                                                  encoding_times_QP6_nRef1_false_false_true[frame_num],
                                                  encoding_times_QP7_nRef1_false_false_true[frame_num],
                                                  encoding_times_QP8_nRef1_false_false_true[frame_num],
                                                  encoding_times_QP9_nRef1_false_false_true[frame_num],
                                                  encoding_times_QP10_nRef1_false_false_true[frame_num]]
        
        # _nRef4_true_true_true [t1, t2, t3, t4, ...]
        encoding_times_nRef4_true_true_true = [encoding_times_QP1_nRef4_true_true_true[frame_num],
                                                  encoding_times_QP2_nRef4_true_true_true[frame_num],
                                                  encoding_times_QP3_nRef4_true_true_true[frame_num],
                                                  encoding_times_QP4_nRef4_true_true_true[frame_num],
                                                  encoding_times_QP5_nRef4_true_true_true[frame_num],
                                                  encoding_times_QP6_nRef4_true_true_true[frame_num],
                                                  encoding_times_QP7_nRef4_true_true_true[frame_num],
                                                  encoding_times_QP8_nRef4_true_true_true[frame_num],
                                                  encoding_times_QP9_nRef4_true_true_true[frame_num],
                                                  encoding_times_QP10_nRef4_true_true_true[frame_num]]
        
        save_curves_plot(QPs, encoding_times_nRef1_false_false_false, QPs, encoding_times_nRef4_false_false_false,
                         QPs, encoding_times_nRef1_true_false_false, QPs, encoding_times_nRef1_false_true_false,
                         QPs, encoding_times_nRef1_false_false_true, QPs, encoding_times_nRef4_true_true_true,
                         "Encoding_times_plots", f'Encoding_time_plot_Frame_{frame_num}', "QP", "encoding time(s)", curve_names)
    
    # Average encoding_time VS QP plot
    # _nRef1_false_false_false_avg [t1, t2, t3, t4, ...]
    encoding_times_nRef1_false_false_false_avg = [np.mean(encoding_times_QP1_nRef1_false_false_false),
                                                  np.mean(encoding_times_QP2_nRef1_false_false_false),
                                                  np.mean(encoding_times_QP3_nRef1_false_false_false),
                                                  np.mean(encoding_times_QP4_nRef1_false_false_false),
                                                  np.mean(encoding_times_QP5_nRef1_false_false_false),
                                                  np.mean(encoding_times_QP6_nRef1_false_false_false),
                                                  np.mean(encoding_times_QP7_nRef1_false_false_false),
                                                  np.mean(encoding_times_QP8_nRef1_false_false_false),
                                                  np.mean(encoding_times_QP9_nRef1_false_false_false),
                                                  np.mean(encoding_times_QP10_nRef1_false_false_false)]
    
    # _nRef4_false_false_false_avg [t1, t2, t3, t4, ...]
    encoding_times_nRef4_false_false_false_avg = [np.mean(encoding_times_QP1_nRef4_false_false_false),
                                                  np.mean(encoding_times_QP2_nRef4_false_false_false),
                                                  np.mean(encoding_times_QP3_nRef4_false_false_false),
                                                  np.mean(encoding_times_QP4_nRef4_false_false_false),
                                                  np.mean(encoding_times_QP5_nRef4_false_false_false),
                                                  np.mean(encoding_times_QP6_nRef4_false_false_false),
                                                  np.mean(encoding_times_QP7_nRef4_false_false_false),
                                                  np.mean(encoding_times_QP8_nRef4_false_false_false),
                                                  np.mean(encoding_times_QP9_nRef4_false_false_false),
                                                  np.mean(encoding_times_QP10_nRef4_false_false_false)]
    
    # _nRef1_true_false_false_avg [t1, t2, t3, t4, ...]
    encoding_times_nRef1_true_false_false_avg = [np.mean(encoding_times_QP1_nRef1_true_false_false),
                                                  np.mean(encoding_times_QP2_nRef1_true_false_false),
                                                  np.mean(encoding_times_QP3_nRef1_true_false_false),
                                                  np.mean(encoding_times_QP4_nRef1_true_false_false),
                                                  np.mean(encoding_times_QP5_nRef1_true_false_false),
                                                  np.mean(encoding_times_QP6_nRef1_true_false_false),
                                                  np.mean(encoding_times_QP7_nRef1_true_false_false),
                                                  np.mean(encoding_times_QP8_nRef1_true_false_false),
                                                  np.mean(encoding_times_QP9_nRef1_true_false_false),
                                                  np.mean(encoding_times_QP10_nRef1_true_false_false)]
    
    # _nRef1_false_true_false_avg [t1, t2, t3, t4, ...]
    encoding_times_nRef1_false_true_false_avg = [np.mean(encoding_times_QP1_nRef1_false_true_false),
                                                  np.mean(encoding_times_QP2_nRef1_false_true_false),
                                                  np.mean(encoding_times_QP3_nRef1_false_true_false),
                                                  np.mean(encoding_times_QP4_nRef1_false_true_false),
                                                  np.mean(encoding_times_QP5_nRef1_false_true_false),
                                                  np.mean(encoding_times_QP6_nRef1_false_true_false),
                                                  np.mean(encoding_times_QP7_nRef1_false_true_false),
                                                  np.mean(encoding_times_QP8_nRef1_false_true_false),
                                                  np.mean(encoding_times_QP9_nRef1_false_true_false),
                                                  np.mean(encoding_times_QP10_nRef1_false_true_false)]
    
    # _nRef1_false_false_true_avg [t1, t2, t3, t4, ...]
    encoding_times_nRef1_false_false_true_avg = [np.mean(encoding_times_QP1_nRef1_false_false_true),
                                                  np.mean(encoding_times_QP2_nRef1_false_false_true),
                                                  np.mean(encoding_times_QP3_nRef1_false_false_true),
                                                  np.mean(encoding_times_QP4_nRef1_false_false_true),
                                                  np.mean(encoding_times_QP5_nRef1_false_false_true),
                                                  np.mean(encoding_times_QP6_nRef1_false_false_true),
                                                  np.mean(encoding_times_QP7_nRef1_false_false_true),
                                                  np.mean(encoding_times_QP8_nRef1_false_false_true),
                                                  np.mean(encoding_times_QP9_nRef1_false_false_true),
                                                  np.mean(encoding_times_QP10_nRef1_false_false_true)]
    
    # _nRef4_true_true_true_avg [t1, t2, t3, t4, ...]
    encoding_times_nRef4_true_true_true_avg = [np.mean(encoding_times_QP1_nRef4_true_true_true),
                                                  np.mean(encoding_times_QP2_nRef4_true_true_true),
                                                  np.mean(encoding_times_QP3_nRef4_true_true_true),
                                                  np.mean(encoding_times_QP4_nRef4_true_true_true),
                                                  np.mean(encoding_times_QP5_nRef4_true_true_true),
                                                  np.mean(encoding_times_QP6_nRef4_true_true_true),
                                                  np.mean(encoding_times_QP7_nRef4_true_true_true),
                                                  np.mean(encoding_times_QP8_nRef4_true_true_true),
                                                  np.mean(encoding_times_QP9_nRef4_true_true_true),
                                                  np.mean(encoding_times_QP10_nRef4_true_true_true)]
    
    save_curves_plot(QPs, encoding_times_nRef1_false_false_false_avg, QPs, encoding_times_nRef4_false_false_false_avg,
                     QPs, encoding_times_nRef1_true_false_false_avg, QPs, encoding_times_nRef1_false_true_false_avg,
                     QPs, encoding_times_nRef1_false_false_true_avg, QPs, encoding_times_nRef4_true_true_true_avg,
                     "Encoding_times_plots", "Average_encoding_time_per_frame", "QP", "encoding time(s)", curve_names)
    

    # Plot decoding time VS QP plots
    QPs = [1,2,3,4,5,6,7,8,9,10]
    for frame_num in range(frame_nums):
        # _nRef1_false_false_false [t1, t2, t3, t4, ...]
        decoding_times_nRef1_false_false_false = [decoding_times_QP1_nRef1_false_false_false[frame_num],
                                                  decoding_times_QP2_nRef1_false_false_false[frame_num],
                                                  decoding_times_QP3_nRef1_false_false_false[frame_num],
                                                  decoding_times_QP4_nRef1_false_false_false[frame_num],
                                                  decoding_times_QP5_nRef1_false_false_false[frame_num],
                                                  decoding_times_QP6_nRef1_false_false_false[frame_num],
                                                  decoding_times_QP7_nRef1_false_false_false[frame_num],
                                                  decoding_times_QP8_nRef1_false_false_false[frame_num],
                                                  decoding_times_QP9_nRef1_false_false_false[frame_num],
                                                  decoding_times_QP10_nRef1_false_false_false[frame_num]]
        
        # _nRef4_false_false_false [t1, t2, t3, t4, ...]
        decoding_times_nRef4_false_false_false = [decoding_times_QP1_nRef4_false_false_false[frame_num],
                                                  decoding_times_QP2_nRef4_false_false_false[frame_num],
                                                  decoding_times_QP3_nRef4_false_false_false[frame_num],
                                                  decoding_times_QP4_nRef4_false_false_false[frame_num],
                                                  decoding_times_QP5_nRef4_false_false_false[frame_num],
                                                  decoding_times_QP6_nRef4_false_false_false[frame_num],
                                                  decoding_times_QP7_nRef4_false_false_false[frame_num],
                                                  decoding_times_QP8_nRef4_false_false_false[frame_num],
                                                  decoding_times_QP9_nRef4_false_false_false[frame_num],
                                                  decoding_times_QP10_nRef4_false_false_false[frame_num]]
        
        # _nRef1_true_false_false [t1, t2, t3, t4, ...]
        decoding_times_nRef1_true_false_false = [decoding_times_QP1_nRef1_true_false_false[frame_num],
                                                 decoding_times_QP2_nRef1_true_false_false[frame_num],
                                                 decoding_times_QP3_nRef1_true_false_false[frame_num],
                                                 decoding_times_QP4_nRef1_true_false_false[frame_num],
                                                 decoding_times_QP5_nRef1_true_false_false[frame_num],
                                                 decoding_times_QP6_nRef1_true_false_false[frame_num],
                                                 decoding_times_QP7_nRef1_true_false_false[frame_num],
                                                 decoding_times_QP8_nRef1_true_false_false[frame_num],
                                                 decoding_times_QP9_nRef1_true_false_false[frame_num],
                                                 decoding_times_QP10_nRef1_true_false_false[frame_num]]
        
        # _nRef1_false_true_false [t1, t2, t3, t4, ...]
        decoding_times_nRef1_false_true_false = [decoding_times_QP1_nRef1_false_true_false[frame_num],
                                                 decoding_times_QP2_nRef1_false_true_false[frame_num],
                                                 decoding_times_QP3_nRef1_false_true_false[frame_num],
                                                 decoding_times_QP4_nRef1_false_true_false[frame_num],
                                                 decoding_times_QP5_nRef1_false_true_false[frame_num],
                                                 decoding_times_QP6_nRef1_false_true_false[frame_num],
                                                 decoding_times_QP7_nRef1_false_true_false[frame_num],
                                                 decoding_times_QP8_nRef1_false_true_false[frame_num],
                                                 decoding_times_QP9_nRef1_false_true_false[frame_num],
                                                 decoding_times_QP10_nRef1_false_true_false[frame_num]]
        
        # _nRef1_false_false_true [t1, t2, t3, t4, ...]
        decoding_times_nRef1_false_false_true = [decoding_times_QP1_nRef1_false_false_true[frame_num],
                                                 decoding_times_QP2_nRef1_false_false_true[frame_num],
                                                 decoding_times_QP3_nRef1_false_false_true[frame_num],
                                                 decoding_times_QP4_nRef1_false_false_true[frame_num],
                                                 decoding_times_QP5_nRef1_false_false_true[frame_num],
                                                 decoding_times_QP6_nRef1_false_false_true[frame_num],
                                                 decoding_times_QP7_nRef1_false_false_true[frame_num],
                                                 decoding_times_QP8_nRef1_false_false_true[frame_num],
                                                 decoding_times_QP9_nRef1_false_false_true[frame_num],
                                                 decoding_times_QP10_nRef1_false_false_true[frame_num]]
        
        # _nRef4_true_true_true [t1, t2, t3, t4, ...]
        decoding_times_nRef4_true_true_true = [decoding_times_QP1_nRef4_true_true_true[frame_num],
                                               decoding_times_QP2_nRef4_true_true_true[frame_num],
                                               decoding_times_QP3_nRef4_true_true_true[frame_num],
                                               decoding_times_QP4_nRef4_true_true_true[frame_num],
                                               decoding_times_QP5_nRef4_true_true_true[frame_num],
                                               decoding_times_QP6_nRef4_true_true_true[frame_num],
                                               decoding_times_QP7_nRef4_true_true_true[frame_num],
                                               decoding_times_QP8_nRef4_true_true_true[frame_num],
                                               decoding_times_QP9_nRef4_true_true_true[frame_num],
                                               decoding_times_QP10_nRef4_true_true_true[frame_num]]
        
        save_curves_plot(QPs, decoding_times_nRef1_false_false_false, QPs, decoding_times_nRef4_false_false_false,
                         QPs, decoding_times_nRef1_true_false_false, QPs, decoding_times_nRef1_false_true_false,
                         QPs, decoding_times_nRef1_false_false_true, QPs, decoding_times_nRef4_true_true_true,
                         "Decoding_times_plots", f'Decoding_time_plot_Frame_{frame_num}', "QP", "decoding time(s)", curve_names)
    
    # Average decoding_time VS QP plot
    # _nRef1_false_false_false_avg [t1, t2, t3, t4, ...]
    decoding_times_nRef1_false_false_false_avg = [np.mean(decoding_times_QP1_nRef1_false_false_false),
                                                  np.mean(decoding_times_QP2_nRef1_false_false_false),
                                                  np.mean(decoding_times_QP3_nRef1_false_false_false),
                                                  np.mean(decoding_times_QP4_nRef1_false_false_false),
                                                  np.mean(decoding_times_QP5_nRef1_false_false_false),
                                                  np.mean(decoding_times_QP6_nRef1_false_false_false),
                                                  np.mean(decoding_times_QP7_nRef1_false_false_false),
                                                  np.mean(decoding_times_QP8_nRef1_false_false_false),
                                                  np.mean(decoding_times_QP9_nRef1_false_false_false),
                                                  np.mean(decoding_times_QP10_nRef1_false_false_false)]
    
    # _nRef4_false_false_false_avg [t1, t2, t3, t4, ...]
    decoding_times_nRef4_false_false_false_avg = [np.mean(decoding_times_QP1_nRef4_false_false_false),
                                                  np.mean(decoding_times_QP2_nRef4_false_false_false),
                                                  np.mean(decoding_times_QP3_nRef4_false_false_false),
                                                  np.mean(decoding_times_QP4_nRef4_false_false_false),
                                                  np.mean(decoding_times_QP5_nRef4_false_false_false),
                                                  np.mean(decoding_times_QP6_nRef4_false_false_false),
                                                  np.mean(decoding_times_QP7_nRef4_false_false_false),
                                                  np.mean(decoding_times_QP8_nRef4_false_false_false),
                                                  np.mean(decoding_times_QP9_nRef4_false_false_false),
                                                  np.mean(decoding_times_QP10_nRef4_false_false_false)]
    
    # _nRef1_true_false_false_avg [t1, t2, t3, t4, ...]
    decoding_times_nRef1_true_false_false_avg = [np.mean(decoding_times_QP1_nRef1_true_false_false),
                                                 np.mean(decoding_times_QP2_nRef1_true_false_false),
                                                 np.mean(decoding_times_QP3_nRef1_true_false_false),
                                                 np.mean(decoding_times_QP4_nRef1_true_false_false),
                                                 np.mean(decoding_times_QP5_nRef1_true_false_false),
                                                 np.mean(decoding_times_QP6_nRef1_true_false_false),
                                                 np.mean(decoding_times_QP7_nRef1_true_false_false),
                                                 np.mean(decoding_times_QP8_nRef1_true_false_false),
                                                 np.mean(decoding_times_QP9_nRef1_true_false_false),
                                                 np.mean(decoding_times_QP10_nRef1_true_false_false)]
    
    # _nRef1_false_true_false_avg [t1, t2, t3, t4, ...]
    decoding_times_nRef1_false_true_false_avg = [np.mean(decoding_times_QP1_nRef1_false_true_false),
                                                 np.mean(decoding_times_QP2_nRef1_false_true_false),
                                                 np.mean(decoding_times_QP3_nRef1_false_true_false),
                                                 np.mean(decoding_times_QP4_nRef1_false_true_false),
                                                 np.mean(decoding_times_QP5_nRef1_false_true_false),
                                                 np.mean(decoding_times_QP6_nRef1_false_true_false),
                                                 np.mean(decoding_times_QP7_nRef1_false_true_false),
                                                 np.mean(decoding_times_QP8_nRef1_false_true_false),
                                                 np.mean(decoding_times_QP9_nRef1_false_true_false),
                                                 np.mean(decoding_times_QP10_nRef1_false_true_false)]
    
    # _nRef1_false_false_true_avg [t1, t2, t3, t4, ...]
    decoding_times_nRef1_false_false_true_avg = [np.mean(decoding_times_QP1_nRef1_false_false_true),
                                                 np.mean(decoding_times_QP2_nRef1_false_false_true),
                                                 np.mean(decoding_times_QP3_nRef1_false_false_true),
                                                 np.mean(decoding_times_QP4_nRef1_false_false_true),
                                                 np.mean(decoding_times_QP5_nRef1_false_false_true),
                                                 np.mean(decoding_times_QP6_nRef1_false_false_true),
                                                 np.mean(decoding_times_QP7_nRef1_false_false_true),
                                                 np.mean(decoding_times_QP8_nRef1_false_false_true),
                                                 np.mean(decoding_times_QP9_nRef1_false_false_true),
                                                 np.mean(decoding_times_QP10_nRef1_false_false_true)]
    
    # _nRef4_true_true_true_avg [t1, t2, t3, t4, ...]
    decoding_times_nRef4_true_true_true_avg = [np.mean(decoding_times_QP1_nRef4_true_true_true),
                                               np.mean(decoding_times_QP2_nRef4_true_true_true),
                                               np.mean(decoding_times_QP3_nRef4_true_true_true),
                                               np.mean(decoding_times_QP4_nRef4_true_true_true),
                                               np.mean(decoding_times_QP5_nRef4_true_true_true),
                                               np.mean(decoding_times_QP6_nRef4_true_true_true),
                                               np.mean(decoding_times_QP7_nRef4_true_true_true),
                                               np.mean(decoding_times_QP8_nRef4_true_true_true),
                                               np.mean(decoding_times_QP9_nRef4_true_true_true),
                                               np.mean(decoding_times_QP10_nRef4_true_true_true)]
    
    save_curves_plot(QPs, decoding_times_nRef1_false_false_false_avg, QPs, decoding_times_nRef4_false_false_false_avg,
                     QPs, decoding_times_nRef1_true_false_false_avg, QPs, decoding_times_nRef1_false_true_false_avg,
                     QPs, decoding_times_nRef1_false_false_true_avg, QPs, decoding_times_nRef4_true_true_true_avg,
                     "Decoding_times_plots", "Average_decoding_time_per_frame", "QP", "decoding time(s)", curve_names)
    

    # Plot VBS_percentages VS QP plots
    QPs = [1,2,3,4,5,6,7,8,9,10]
    for frame_num in range(frame_nums):
        # _nRef1_false_false_false [p1, p2, p3, p4, ...]
        VBS_percentages_nRef1_false_false_false = [VBS_percentages_QP1_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP2_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP3_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP4_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP5_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP6_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP7_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP8_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP9_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP10_nRef1_false_false_false[frame_num]]
        
        # _nRef4_false_false_false [p1, p2, p3, p4, ...]
        VBS_percentages_nRef4_false_false_false = [VBS_percentages_QP1_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP2_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP3_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP4_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP5_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP6_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP7_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP8_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP9_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP10_nRef4_false_false_false[frame_num]]
        
        # _nRef1_true_false_false [p1, p2, p3, p4, ...]
        VBS_percentages_nRef1_true_false_false = [VBS_percentages_QP1_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP2_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP3_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP4_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP5_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP6_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP7_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP8_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP9_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP10_nRef1_true_false_false[frame_num]]
        
        # _nRef1_false_true_false [p1, p2, p3, p4, ...]
        VBS_percentages_nRef1_false_true_false = [VBS_percentages_QP1_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP2_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP3_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP4_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP5_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP6_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP7_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP8_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP9_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP10_nRef1_false_true_false[frame_num]]
        
        # _nRef1_false_false_true [p1, p2, p3, p4, ...]
        VBS_percentages_nRef1_false_false_true = [VBS_percentages_QP1_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP2_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP3_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP4_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP5_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP6_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP7_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP8_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP9_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP10_nRef1_false_false_true[frame_num]]
        
        # _nRef4_true_true_true [p1, p2, p3, p4, ...]
        VBS_percentages_nRef4_true_true_true = [VBS_percentages_QP1_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP2_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP3_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP4_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP5_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP6_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP7_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP8_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP9_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP10_nRef4_true_true_true[frame_num]]
        
        save_curves_plot(QPs, VBS_percentages_nRef1_false_false_false, QPs, VBS_percentages_nRef4_false_false_false,
                         QPs, VBS_percentages_nRef1_true_false_false, QPs, VBS_percentages_nRef1_false_true_false,
                         QPs, VBS_percentages_nRef1_false_false_true, QPs, VBS_percentages_nRef4_true_true_true,
                         "VBS_percentages_VS_QP_plots", f'VBS_percentages_VS_QP_plot_Frame_{frame_num}', "QP", "VBS percentage", curve_names)
    

    # _nRef1_false_false_false_avg [p1, p2, p3, p4, ...]
    VBS_percentages_nRef1_false_false_false_avg = [np.mean(VBS_percentages_QP1_nRef1_false_false_false),
                                                   np.mean(VBS_percentages_QP2_nRef1_false_false_false),
                                                   np.mean(VBS_percentages_QP3_nRef1_false_false_false),
                                                   np.mean(VBS_percentages_QP4_nRef1_false_false_false),
                                                   np.mean(VBS_percentages_QP5_nRef1_false_false_false),
                                                   np.mean(VBS_percentages_QP6_nRef1_false_false_false),
                                                   np.mean(VBS_percentages_QP7_nRef1_false_false_false),
                                                   np.mean(VBS_percentages_QP8_nRef1_false_false_false),
                                                   np.mean(VBS_percentages_QP9_nRef1_false_false_false),
                                                   np.mean(VBS_percentages_QP10_nRef1_false_false_false)]
    
    # _nRef4_false_false_false_avg [p1, p2, p3, p4, ...]
    VBS_percentages_nRef4_false_false_false_avg = [np.mean(VBS_percentages_QP1_nRef4_false_false_false),
                                                   np.mean(VBS_percentages_QP2_nRef4_false_false_false),
                                                   np.mean(VBS_percentages_QP3_nRef4_false_false_false),
                                                   np.mean(VBS_percentages_QP4_nRef4_false_false_false),
                                                   np.mean(VBS_percentages_QP5_nRef4_false_false_false),
                                                   np.mean(VBS_percentages_QP6_nRef4_false_false_false),
                                                   np.mean(VBS_percentages_QP7_nRef4_false_false_false),
                                                   np.mean(VBS_percentages_QP8_nRef4_false_false_false),
                                                   np.mean(VBS_percentages_QP9_nRef4_false_false_false),
                                                   np.mean(VBS_percentages_QP10_nRef4_false_false_false)]
    
    # _nRef1_true_false_false_avg [p1, p2, p3, p4, ...]
    VBS_percentages_nRef1_true_false_false_avg = [np.mean(VBS_percentages_QP1_nRef1_true_false_false),
                                                  np.mean(VBS_percentages_QP2_nRef1_true_false_false),
                                                  np.mean(VBS_percentages_QP3_nRef1_true_false_false),
                                                  np.mean(VBS_percentages_QP4_nRef1_true_false_false),
                                                  np.mean(VBS_percentages_QP5_nRef1_true_false_false),
                                                  np.mean(VBS_percentages_QP6_nRef1_true_false_false),
                                                  np.mean(VBS_percentages_QP7_nRef1_true_false_false),
                                                  np.mean(VBS_percentages_QP8_nRef1_true_false_false),
                                                  np.mean(VBS_percentages_QP9_nRef1_true_false_false),
                                                  np.mean(VBS_percentages_QP10_nRef1_true_false_false)]
    
    # _nRef1_false_true_false_avg [p1, p2, p3, p4, ...]
    VBS_percentages_nRef1_false_true_false_avg = [np.mean(VBS_percentages_QP1_nRef1_false_true_false),
                                                  np.mean(VBS_percentages_QP2_nRef1_false_true_false),
                                                  np.mean(VBS_percentages_QP3_nRef1_false_true_false),
                                                  np.mean(VBS_percentages_QP4_nRef1_false_true_false),
                                                  np.mean(VBS_percentages_QP5_nRef1_false_true_false),
                                                  np.mean(VBS_percentages_QP6_nRef1_false_true_false),
                                                  np.mean(VBS_percentages_QP7_nRef1_false_true_false),
                                                  np.mean(VBS_percentages_QP8_nRef1_false_true_false),
                                                  np.mean(VBS_percentages_QP9_nRef1_false_true_false),
                                                  np.mean(VBS_percentages_QP10_nRef1_false_true_false)]
    
    # _nRef1_false_false_true_avg [p1, p2, p3, p4, ...]
    VBS_percentages_nRef1_false_false_true_avg = [np.mean(VBS_percentages_QP1_nRef1_false_false_true),
                                                  np.mean(VBS_percentages_QP2_nRef1_false_false_true),
                                                  np.mean(VBS_percentages_QP3_nRef1_false_false_true),
                                                  np.mean(VBS_percentages_QP4_nRef1_false_false_true),
                                                  np.mean(VBS_percentages_QP5_nRef1_false_false_true),
                                                  np.mean(VBS_percentages_QP6_nRef1_false_false_true),
                                                  np.mean(VBS_percentages_QP7_nRef1_false_false_true),
                                                  np.mean(VBS_percentages_QP8_nRef1_false_false_true),
                                                  np.mean(VBS_percentages_QP9_nRef1_false_false_true),
                                                  np.mean(VBS_percentages_QP10_nRef1_false_false_true)]
    
    # _nRef4_true_true_true_avg [p1, p2, p3, p4, ...]
    VBS_percentages_nRef4_true_true_true_avg = [np.mean(VBS_percentages_QP1_nRef4_true_true_true),
                                                np.mean(VBS_percentages_QP2_nRef4_true_true_true),
                                                np.mean(VBS_percentages_QP3_nRef4_true_true_true),
                                                np.mean(VBS_percentages_QP4_nRef4_true_true_true),
                                                np.mean(VBS_percentages_QP5_nRef4_true_true_true),
                                                np.mean(VBS_percentages_QP6_nRef4_true_true_true),
                                                np.mean(VBS_percentages_QP7_nRef4_true_true_true),
                                                np.mean(VBS_percentages_QP8_nRef4_true_true_true),
                                                np.mean(VBS_percentages_QP9_nRef4_true_true_true),
                                                np.mean(VBS_percentages_QP10_nRef4_true_true_true)]
    
    save_curves_plot(QPs, VBS_percentages_nRef1_false_false_false_avg, QPs, VBS_percentages_nRef4_false_false_false_avg,
                     QPs, VBS_percentages_nRef1_true_false_false_avg, QPs, VBS_percentages_nRef1_false_true_false_avg,
                     QPs, VBS_percentages_nRef1_false_false_true_avg, QPs, VBS_percentages_nRef4_true_true_true_avg,
                     "VBS_percentages_VS_QP_plots", "Average_VBS_percentages_VS_QP_plot", "QP", "VBS percentage", curve_names)
    
    # Plot VBS_percentages VS bit_sizes plots
    for frame_num in range(frame_nums):
        # _nRef1_false_false_false [p1, p2, p3, p4, ...]
        bit_size_nRef1_false_false_false = [RDs_QP1_nRef1_false_false_false[frame_num][1],
                                            RDs_QP2_nRef1_false_false_false[frame_num][1],
                                            RDs_QP3_nRef1_false_false_false[frame_num][1],
                                            RDs_QP4_nRef1_false_false_false[frame_num][1],
                                            RDs_QP5_nRef1_false_false_false[frame_num][1],
                                            RDs_QP6_nRef1_false_false_false[frame_num][1],
                                            RDs_QP7_nRef1_false_false_false[frame_num][1],
                                            RDs_QP8_nRef1_false_false_false[frame_num][1],
                                            RDs_QP9_nRef1_false_false_false[frame_num][1],
                                            RDs_QP10_nRef1_false_false_false[frame_num][1]]
        VBS_percentages_nRef1_false_false_false = [VBS_percentages_QP1_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP2_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP3_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP4_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP5_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP6_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP7_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP8_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP9_nRef1_false_false_false[frame_num],
                                                   VBS_percentages_QP10_nRef1_false_false_false[frame_num]]
        
        # _nRef4_false_false_false [p1, p2, p3, p4, ...]
        bit_size_nRef4_false_false_false = [RDs_QP1_nRef4_false_false_false[frame_num][1],
                                            RDs_QP2_nRef4_false_false_false[frame_num][1],
                                            RDs_QP3_nRef4_false_false_false[frame_num][1],
                                            RDs_QP4_nRef4_false_false_false[frame_num][1],
                                            RDs_QP5_nRef4_false_false_false[frame_num][1],
                                            RDs_QP6_nRef4_false_false_false[frame_num][1],
                                            RDs_QP7_nRef4_false_false_false[frame_num][1],
                                            RDs_QP8_nRef4_false_false_false[frame_num][1],
                                            RDs_QP9_nRef4_false_false_false[frame_num][1],
                                            RDs_QP10_nRef4_false_false_false[frame_num][1]]
        VBS_percentages_nRef4_false_false_false = [VBS_percentages_QP1_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP2_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP3_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP4_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP5_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP6_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP7_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP8_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP9_nRef4_false_false_false[frame_num],
                                                   VBS_percentages_QP10_nRef4_false_false_false[frame_num]]
        
        # _nRef1_true_false_false [p1, p2, p3, p4, ...]
        bit_size_nRef1_true_false_false = [RDs_QP1_nRef1_true_false_false[frame_num][1],
                                            RDs_QP2_nRef1_true_false_false[frame_num][1],
                                            RDs_QP3_nRef1_true_false_false[frame_num][1],
                                            RDs_QP4_nRef1_true_false_false[frame_num][1],
                                            RDs_QP5_nRef1_true_false_false[frame_num][1],
                                            RDs_QP6_nRef1_true_false_false[frame_num][1],
                                            RDs_QP7_nRef1_true_false_false[frame_num][1],
                                            RDs_QP8_nRef1_true_false_false[frame_num][1],
                                            RDs_QP9_nRef1_true_false_false[frame_num][1],
                                            RDs_QP10_nRef1_true_false_false[frame_num][1]]
        VBS_percentages_nRef1_true_false_false = [VBS_percentages_QP1_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP2_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP3_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP4_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP5_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP6_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP7_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP8_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP9_nRef1_true_false_false[frame_num],
                                                  VBS_percentages_QP10_nRef1_true_false_false[frame_num]]
        
        # _nRef1_false_true_false [p1, p2, p3, p4, ...]
        bit_size_nRef1_false_true_false = [RDs_QP1_nRef1_false_true_false[frame_num][1],
                                            RDs_QP2_nRef1_false_true_false[frame_num][1],
                                            RDs_QP3_nRef1_false_true_false[frame_num][1],
                                            RDs_QP4_nRef1_false_true_false[frame_num][1],
                                            RDs_QP5_nRef1_false_true_false[frame_num][1],
                                            RDs_QP6_nRef1_false_true_false[frame_num][1],
                                            RDs_QP7_nRef1_false_true_false[frame_num][1],
                                            RDs_QP8_nRef1_false_true_false[frame_num][1],
                                            RDs_QP9_nRef1_false_true_false[frame_num][1],
                                            RDs_QP10_nRef1_false_true_false[frame_num][1]]
        VBS_percentages_nRef1_false_true_false = [VBS_percentages_QP1_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP2_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP3_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP4_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP5_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP6_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP7_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP8_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP9_nRef1_false_true_false[frame_num],
                                                  VBS_percentages_QP10_nRef1_false_true_false[frame_num]]
        
        # _nRef1_false_false_true [p1, p2, p3, p4, ...]
        bit_size_nRef1_false_false_true = [RDs_QP1_nRef1_false_false_true[frame_num][1],
                                            RDs_QP2_nRef1_false_false_true[frame_num][1],
                                            RDs_QP3_nRef1_false_false_true[frame_num][1],
                                            RDs_QP4_nRef1_false_false_true[frame_num][1],
                                            RDs_QP5_nRef1_false_false_true[frame_num][1],
                                            RDs_QP6_nRef1_false_false_true[frame_num][1],
                                            RDs_QP7_nRef1_false_false_true[frame_num][1],
                                            RDs_QP8_nRef1_false_false_true[frame_num][1],
                                            RDs_QP9_nRef1_false_false_true[frame_num][1],
                                            RDs_QP10_nRef1_false_false_true[frame_num][1]]
        VBS_percentages_nRef1_false_false_true = [VBS_percentages_QP1_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP2_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP3_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP4_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP5_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP6_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP7_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP8_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP9_nRef1_false_false_true[frame_num],
                                                  VBS_percentages_QP10_nRef1_false_false_true[frame_num]]
        
        # _nRef4_true_true_true [p1, p2, p3, p4, ...]
        bit_size_nRef4_true_true_true = [RDs_QP1_nRef4_true_true_true[frame_num][1],
                                            RDs_QP2_nRef4_true_true_true[frame_num][1],
                                            RDs_QP3_nRef4_true_true_true[frame_num][1],
                                            RDs_QP4_nRef4_true_true_true[frame_num][1],
                                            RDs_QP5_nRef4_true_true_true[frame_num][1],
                                            RDs_QP6_nRef4_true_true_true[frame_num][1],
                                            RDs_QP7_nRef4_true_true_true[frame_num][1],
                                            RDs_QP8_nRef4_true_true_true[frame_num][1],
                                            RDs_QP9_nRef4_true_true_true[frame_num][1],
                                            RDs_QP10_nRef4_true_true_true[frame_num][1]]
        VBS_percentages_nRef4_true_true_true = [VBS_percentages_QP1_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP2_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP3_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP4_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP5_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP6_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP7_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP8_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP9_nRef4_true_true_true[frame_num],
                                                VBS_percentages_QP10_nRef4_true_true_true[frame_num]]
        
        save_curves_plot(bit_size_nRef1_false_false_false, VBS_percentages_nRef1_false_false_false, bit_size_nRef4_false_false_false, VBS_percentages_nRef4_false_false_false,
                         bit_size_nRef1_true_false_false, VBS_percentages_nRef1_true_false_false, bit_size_nRef1_false_true_false, VBS_percentages_nRef1_false_true_false,
                         bit_size_nRef1_false_false_true, VBS_percentages_nRef1_false_false_true, bit_size_nRef4_true_true_true, VBS_percentages_nRef4_true_true_true,
                         "VBS_percentages_VS_bit_sizes_plots", f'VBS_percentages_VS_bit_sizes_plot_Frame_{frame_num}', "bit size (in bits)", "VBS percentage", curve_names)
    
    
    save_curves_plot(bit_size_nRef1_false_false_false_avg, VBS_percentages_nRef1_false_false_false_avg, bit_size_nRef4_false_false_false_avg, VBS_percentages_nRef4_false_false_false_avg,
                     bit_size_nRef1_true_false_false_avg, VBS_percentages_nRef1_true_false_false_avg, bit_size_nRef1_false_true_false_avg, VBS_percentages_nRef1_false_true_false_avg,
                     bit_size_nRef1_false_false_true_avg, VBS_percentages_nRef1_false_false_true_avg, bit_size_nRef4_true_true_true_avg, VBS_percentages_nRef4_true_true_true_avg,
                     "VBS_percentages_VS_bit_sizes_plots", "Average_VBS_percentages_VS_bit_sizes_plot", "QP", "VBS percentage", curve_names)
plot_main()