import sys
# sys.path.append('/home/justin/Documents/GitHub/giftolexia/basicivt/src')

import ntpath
from os import listdir, path
from os.path import isfile, join, exists
import datetime
import glob
import numpy as np
import pandas as pd

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import PupilIVT as ivt


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def path_path(path):
    head, tail = ntpath.split(path)
    return head


def find_data_files(path="*.csv"):
    if (len(path) < 1):
        path = path + "*.csv"

    if (path != "*.csv"):
        path = path.replace('\\', '/')  # Convert windows paths to unix paths

    if (path[len(path) - 1] == "/"):
        path = path + "*.csv"

    datafiles = glob.glob(path)
    return datafiles


# CSV file contains repeated timestamps. To avoid this, timestamps are calculated and re-assigned
# Since the frequency is 40Hz, samples are taken at every 40 milliseconds. Hence, each timestamp
# is a multiple of the record serial number with 40.
def assign_incremental_timestamps(df, sample_freq):
    df["slno"] = df.index
    df["TimeStamp"] = df["slno"] * sample_freq  # Time in milliseconds
    df["TimeMillis"] = df["slno"] * sample_freq / 1000  # Convert to Seconds
    return df


# Wrapper to call all pre-processing to be done on the input raw data
def preprocess_data(df, sample_freq):
    print("INFO: preprocess_data: Assign timestamps incrementally. (step by " + str(sample_freq) + "ms)")
    df = assign_incremental_timestamps(df, sample_freq)
    return df


# Extract the path and the file name, without the extension
def extract_file_wo_extn(file):
    path = path_path(file)  # separate path
    file = path_leaf(file)  # separate file name

    # calculate the filename sans extension
    file_wo_extn = "noname"
    if (len(file) > 0):
        file_wo_extn = file.split(".")[0]

    if (len(path) > 0):
        file_wo_extn = path + "/" + file_wo_extn

    return file_wo_extn


# Prepare the graph image file name. The file name and path will be the same as csv file, except that the
# file extension will be replaced by .png
def prepare_graph_filename(file):
    imgfile = ""
    imgfile = extract_file_wo_extn(file) + ".png"
    return imgfile


# prepare file name to store ivt intermediate results - csv file
def prepare_intermediate_data_filename(file, tag="left"):
    interfile = ""
    interfile = extract_file_wo_extn(file) + "_ivtdata_" + tag + ".csv"
    return interfile


# Generate all charts for a data file and save them as png image
def generate_graphs(data, data_df, l_fix, r_fix, fs_left_df, fs_right_df, file):
    imgfile = prepare_graph_filename(file)
    data_zoomedin = data_df.query("SubTest < 1 ")  # Subset the data for first 1 second
    print("INFO: generate_graphs: Saving graph to " + imgfile)

    fig, ax = plt.subplots(figsize=(18, 20), ncols=2, nrows=5)
    fig.suptitle(file + "- Pupil Movements", y=1.09, fontsize=20)
    fig.subplots_adjust(hspace=0.3)

    ax[0][0].set_title('Left Pupil X (Full Test)')
    ax[0][0].plot(data["TimeStamp"], data["LeftPupil.X"], color='red')
    ax[0][0].set(xlabel='TimeStamp', ylabel='X Movement')

    ax[0][1].set_title('Right Pupil X (Full Test)')
    ax[0][1].plot(data["TimeStamp"], data["RightPupil.X"], color='red')
    ax[0][1].set(xlabel='TimeStamp', ylabel='X Movement')

    ax[1][0].set_title('Left Pupil X (Zoomed In - SubTest=1)')
    ax[1][0].plot(data_zoomedin["TimeStamp"], data_zoomedin["LeftPupil.X"], color='red')
    ax[1][0].set(xlabel='TimeStamp', ylabel='X Movement')

    ax[1][1].set_title('Right Pupil X (Zoomed In - SubTest=1)')
    ax[1][1].plot(data_zoomedin["TimeStamp"], data_zoomedin["RightPupil.X"], color='red')
    ax[1][1].set(xlabel='TimeStamp', ylabel='X Movement')

    ax[2][0].set_title('Left Pupil Y')
    ax[2][0].plot(data["TimeStamp"], data["LeftPupil.Y"], color='blue')
    ax[2][0].set(xlabel='TimeStamp', ylabel='Y Movement')

    ax[2][1].set_title('Right Pupil Y')
    ax[2][1].plot(data["TimeStamp"], data["RightPupil.Y"], color='blue')
    ax[2][1].set(xlabel='TimeStamp', ylabel='Y Movement')

    ax[3][0].set_title('Left Pupil Fixation Distribution')
    # ax[2][0].hist(l_fix["duration"])
    sns.distplot(l_fix["duration"], hist=True, rug=True, ax=ax[3][0])
    ax[3][0].set(xlabel='Fixation Duration', ylabel='Density')
    ax[3][0].set_xticks(np.arange(0, 2, 0.1))

    ax[3][1].set_title('Right Pupil Fixation Distribution')
    # ax[2][1].hist(r_fix["duration"])
    sns.distplot(r_fix["duration"], hist=True, rug=True, ax=ax[3][1])
    ax[3][1].set(xlabel='Fixation Duration', ylabel='Density')
    ax[3][1].set_xticks(np.arange(0, 2, 0.1))

    ax[4][0].set_title('Left Pupil Fixation Histogram')
    ax[4][0].hist(l_fix["duration"])
    ax[4][0].set(xlabel='Fixation Duration', ylabel='Frequency')
    ax[4][0].set_xticks(np.arange(0, 2, 0.1))

    ax[4][1].set_title('Right Pupil Fixation Histogram')
    ax[4][1].hist(r_fix["duration"])
    ax[4][1].set(xlabel='Fixation Duration', ylabel='Frequency')
    ax[4][1].set_xticks(np.arange(0, 2, 0.1))

    fig.savefig(imgfile, dpi=fig.dpi, bbox_inches='tight')
    return 1


# Get the file path from the commandline parameters
# User can specify either a csv file name or the path containing multiple csv files.
# If a directory is specified, it must end with a "/".
def get_cmdline_param_path(args):
    path = "./"  # Default to current path
    if (len(args) > 1):  # atleast one parameter specified
        path = args[1]
    return path


#
# The set of markers identified after IVT and further processing for fixations and saccades are to be stored
# into a common file, where all kids data accumulate. Each row in the marker csv file is identified with the
# eyetracker data file name. Also, a timestamp column (y-m-d) is added for audit purposes.
#
def store_markers(markers, source_file, marker_columns, marker_csv_file):
    marker_field_sep = ","
    marker_headers = True
    # marker_columns = ["mfl_mean_fixation_length","fixation_duration_median","fixation_duration_mode","fixation_duration_sd","total_fixation_count","total_fwd_saccade_count","total_rev_saccade_count","total_time_taken","total_sub_tests","time_stamp", "file"]

    new_markers_df = pd.DataFrame([markers])

    # If markers file exists in current directory, read the file into a data frame, append the new record.
    # and save it back.
    # Else, create the csv file and add the new record
    try:
        if (path.exists(marker_csv_file)):
            print("INFO: marker_csv_file found in the current directory. Appending new markers.")
            markers_hist_df = pd.read_csv(marker_csv_file, sep=marker_field_sep, header=0)
            markers_concat_df = pd.concat([markers_hist_df[marker_columns], new_markers_df[marker_columns]])
            markers_concat_df.to_csv(marker_csv_file, sep=marker_field_sep, header=marker_headers, index=False)
        else:
            print("INFO: marker_csv_file not found in the current directory. Creating.")
            new_markers_df[marker_columns].to_csv(marker_csv_file, sep=marker_field_sep, header=marker_headers,
                                                  index=False)
        print("INFO: Added new markers to " + marker_csv_file + " in current directory")
    except Exception as e:
        print("ERROR: Could not read/write " + marker_csv_file + " in current directory")
        print(e)
        return False


#
# Set of functions that calculates various markers derived from the source data and IVT processed data
#
def calc_mean_fixation_duration(fs_left_df, fs_right_df):
    # Marker: MFL - Mean of ﬁxation lengths: Summation of the lenght in time of ﬁxations divided by the number of ﬁxations
    # Query for fixations in the left and right pupil arrays, calculate left and right pupil averages and then
    # calculate the final avg duration as the average of left and right eyes.
    #
    l_avg_fixation_duration = fs_left_df.query("action=='F'")["duration"].mean()
    r_avg_fixation_duration = fs_right_df.query("action=='F'")["duration"].mean()
    avg_fixation_duration = (l_avg_fixation_duration + r_avg_fixation_duration) / 2
    return avg_fixation_duration


def calc_median_fixation_duration(fs_left_df, fs_right_df):
    #
    # Query for fixations in the left and right pupil arrays, calculate left and right pupil medians and then
    # calculate the final median duration as the average of left and right eye medians.
    #
    l_med_fixation_duration = fs_left_df.query("action=='F'")["duration"].median()
    r_med_fixation_duration = fs_right_df.query("action=='F'")["duration"].median()
    median_fixation_duration = (l_med_fixation_duration + r_med_fixation_duration) / 2
    return median_fixation_duration


def calc_mode_fixation_duration(fs_left_df, fs_right_df):
    #
    # Query for fixations in the left and right pupil arrays, calculate left and right pupil modes and then
    # calculate the final mode duration as the average of left and right eye mode values.
    #
    l_mod_fixation_duration = fs_left_df.query("action=='F'")["duration"].mode().mean()
    r_mod_fixation_duration = fs_right_df.query("action=='F'")["duration"].mode().mean()
    mode_fixation_duration = (l_mod_fixation_duration + r_mod_fixation_duration) / 2
    return mode_fixation_duration


def calc_sd_fixation_duration(fs_left_df, fs_right_df):
    #
    # Query for fixations in the left and right pupil arrays, calculate standard deviation of fixation durations
    # for left and right pupil and then calculate the final SD as the average of left and right eyes
    #
    l_sd_fixation_duration = fs_left_df.query("action=='F'")["duration"].std()
    r_sd_fixation_duration = fs_right_df.query("action=='F'")["duration"].std()
    sd_fixation_duration = (l_sd_fixation_duration + r_sd_fixation_duration) / 2
    return sd_fixation_duration


def calc_total_fixation_count(fs_left_df, fs_right_df):
    # Calculate the number of fixations for each eye separately and then average them out
    left_count = 0
    right_count = 0
    if (fs_left_df is not None):
        left_count = len(fs_left_df.query("action=='F'"))

    if (fs_right_df is not None):
        right_count = len(fs_right_df.query("action=='F'"))

    total_fixation_count = (left_count + right_count) / 2
    return total_fixation_count


# Calculate total saccades occurred for both eyes and averages them out.
# saccade_direction can be : F - Forward or R - Reverse
def calc_total_saccade_count(fs_left_df, fs_right_df, saccade_direction='F'):
    # Calculate the number of saccades for each eye separately and then average them out
    left_count = 0
    right_count = 0
    if (fs_left_df is not None):
        left_count = len(fs_left_df.query("action=='S' and saccade_direction=='" + saccade_direction + "'"))

    if (fs_right_df is not None):
        right_count = len(fs_right_df.query("action=='S' and saccade_direction=='" + saccade_direction + "'"))

    total_saccade_count = (left_count + right_count) // 2
    return total_saccade_count


def calc_total_time_taken(data_df):
    # Total time taken by the child to finish reading
    total_time_taken = 0
    if (data_df is not None):
        total_time_taken = data_df["TimeMillis"].max()

    return total_time_taken


def calc_total_subtest_count(data_df):
    # Count of sub-tests within the input data - an indicator of the length of test
    total_subtest_count = 0
    if (data_df is not None):
        total_subtest_count = data_df["SubTest"].nunique()
    return total_subtest_count


def calc_total_data_points(fs_left_df, fs_right_df, data_df):
    # calculate the total number of data points, after merging adjacent fixations and saccades.

    total_data_points = 0
    # left_data_points = len (fs_left_df)
    # right_data_points = len (fs_right_df)
    # total_data_points = (left_data_points + right_data_points) / 2
    total_data_points = len(data_df)
    return total_data_points


def calc_fixation_frequency(fs_left_df, fs_right_df, data_df):
    # Marker FF: Fixation Frequency = Total number of ﬁxations divided by the total number of data points for the reading.
    total_fixation_count = calc_total_fixation_count(fs_left_df, fs_right_df)
    print("total fixation count calculated: {0}".format(total_fixation_count))

    total_data_points = calc_total_data_points(fs_left_df, fs_right_df, data_df)
    print("total data points calculated: {0}".format(total_data_points))
    if (total_data_points < 1):
        return 0
    return total_fixation_count / total_data_points


def calc_fwd_saccade_frequency(fs_left_df, fs_right_df, data_df):
    # Marker FF: Fixation Frequency = Total number of ﬁxations divided by the total number of data points for the reading.
    total_fwd_saccade_count = calc_total_saccade_count(fs_left_df, fs_right_df, saccade_direction='F')
    total_data_points = calc_total_data_points(fs_left_df, fs_right_df, data_df)

    if (total_data_points < 1):
        return 0
    return total_fwd_saccade_count / total_data_points


def calc_fixation_stability(data_df):
    # Marker FS: Fixation stability: Mean of distance of movement within a ﬁxation.
    left_fix_stability = data_df.query("fixation_left == 1")["distance_left"].mean()
    right_fix_stability = data_df.query("fixation_right == 1")["distance_right"].mean()

    fix_stability = (left_fix_stability + right_fix_stability) / 2
    return fix_stability


def calc_saccadic_amplitude(data_df):
    # Marker SA: Saccadic amplitude: Mean of the speed(distance divided by time taken) of saccades
    left_sa = data_df.query("fixation_left == 0")["velocity_left"].mean()
    right_sa = data_df.query("fixation_left == 0")["velocity_right"].mean()

    sa = (left_sa + right_sa) / 2
    return sa


def calc_regression_saccade_frequency(fs_left_df, fs_right_df, data_df):
    # Marker RF: Regression/Backward saccade frequency: Total number of backward saccades divided by the total number of data points for the reading
    total_rev_saccade_count = calc_total_saccade_count(fs_left_df, fs_right_df, saccade_direction='R')
    total_data_points = calc_total_data_points(fs_left_df, fs_right_df, data_df)
    if (total_data_points < 1):
        return 0
    return total_rev_saccade_count / total_data_points


def calc_mean_fwd_saccade_length(fs_left_df, fs_right_df):
    # Marker MFS: Mean of forward saccade lengths: Summation of the distance of all forward saccades divided by the number of forward saccades.
    mean_fwd_saccade_length = 0

    fwd_saccades_left_distance = fs_left_df.query("(action=='S') & (saccade_direction == 'F') ")["distance"].sum()
    fwd_saccades_right_distance = fs_right_df.query("(action=='S') & (saccade_direction == 'F') ")["distance"].sum()

    fwd_saccades_distance_avg = (fwd_saccades_left_distance + fwd_saccades_right_distance) / 2

    total_fwd_saccade_count = calc_total_saccade_count(fs_left_df, fs_right_df, saccade_direction='F')

    if (total_fwd_saccade_count != 0):
        mean_fwd_saccade_length = fwd_saccades_distance_avg / total_fwd_saccade_count

    return mean_fwd_saccade_length


def main(args):
    markers = {}
    marker_columns = []

    markers_file = "markers.csv"
    # get the path to csv files. If not specified in the commandline, it will be defaulted to "./" (current directory)
    mypath = get_cmdline_param_path(args)

    # mypath = "/commonarea/vboxshared/datascience/slam/lexia/testdata/Benjamin.csv"
    # mypath = "/commonarea/vboxshared/datascience/slam/lexia/testdata/Laasya.csv"

    sample_freq = 40  # 25 Hz (time between samples in milliseconds)
    IVT_VELOCITY_THRESHOLD = 20

    datafiles = find_data_files(path=mypath)

    if (len(datafiles) == 0):
        print("WARN: No csv files found in " + mypath)
        return 0

    print("INFO: Found " + str(len(datafiles)) + " files at " + mypath)

    for file in datafiles:
        print("INFO: Processing file: " + file)
        data = pd.read_csv(file, sep=",", header=0)
        if (data is None):
            print("ERROR: Could not read " + file)

        print("INFO: Read " + str(data.shape[0]) + " records from: " + file)

        data = preprocess_data(data, sample_freq)
        #
        # Call IVT functions to mark fixations and saccade rows
        #
        ivtobj = ivt.PupilIVT(sample_freq=40, threshold_velocity=IVT_VELOCITY_THRESHOLD)
        data_df = ivtobj.calc_horizontal_velocity(data)
        # Merge adjacent fixations and saccades for left eye
        fs_left_df = ivtobj.process_fixation_saccades(data_df, tag="left")
        # Merge adjacent fixations and saccades for right eye
        fs_right_df = ivtobj.process_fixation_saccades(data_df, tag="right")
        # Mark saccades as forward or reverse for left eye
        fs_left_df = ivtobj.calc_saccade_direction(data_df, fs_left_df, tag="left")
        # Mark saccades as forward or reverse for right eye
        fs_right_df = ivtobj.calc_saccade_direction(data_df, fs_right_df, tag="right")

        # Storing intermediate data processed by IVT algorithm.
        # may be used for fine tuning and debugging purposes.
        # One file each for left and right eye. Named after <original_file>_ivtdata_<left/right>.csv
        #
        left_ivt_data_file = prepare_intermediate_data_filename(file, tag="left")
        print("INFO: Storing IVT (intermediate data, left eye ) file: " + left_ivt_data_file)
        # fs_left_df.to_csv(left_ivt_data_file, sep=",", header=True, index=False)

        right_ivt_data_file = prepare_intermediate_data_filename(file, tag="right")
        print("INFO: Storing IVT (intermediate data, right eye ) file: " + right_ivt_data_file)
        # fs_right_df.to_csv(right_ivt_data_file, sep=",", header=True, index=False)

        calc_data_file = prepare_intermediate_data_filename(file, tag="calc")
        print("INFO: Storing IVT (intermediate data, calculations ) file: " + calc_data_file)
        data_df.to_csv(calc_data_file, sep=",", header=True, index=False)

        #
        # Process resultant dataframes (fs_left_df, fs_right_df and data_df as necessary)
        # and prepare markers
        # MFL - Mean of ﬁxation lengths: Summation of the length in time of ﬁxations divided by the number of ﬁxations.
        marker_name = "mfl_mean_fixation_length"
        mean_fixation_duration = calc_mean_fixation_duration(fs_left_df, fs_right_df)  # change
        markers[marker_name] = round(mean_fixation_duration, 3)
        marker_columns.append(marker_name)
        print("INFO: calculated mfl_mean_fixation_length")

        marker_name = "fixation_duration_median"
        med_fixation_duration = calc_median_fixation_duration(fs_left_df, fs_right_df)
        markers[marker_name] = round(med_fixation_duration, 3)
        marker_columns.append(marker_name)
        print("INFO: calculated fixation_duration_median")

        marker_name = "fixation_duration_mode"
        mode_fixation_duration = calc_mode_fixation_duration(fs_left_df, fs_right_df)
        markers[marker_name] = round(mode_fixation_duration, 3)
        marker_columns.append(marker_name)
        print("INFO: calculated fixation_duration_mode")

        marker_name = "fixation_duration_sd"
        sd_fixation_duration = calc_sd_fixation_duration(fs_left_df, fs_right_df)
        markers[marker_name] = round(sd_fixation_duration, 3)
        marker_columns.append(marker_name)
        print("INFO: calculated fixation_duration_sd")

        marker_name = "total_fixation_count"
        total_fixation_count = calc_total_fixation_count(fs_left_df, fs_right_df)
        markers[marker_name] = int(total_fixation_count)
        marker_columns.append(marker_name)
        print("INFO: calculated total_fixation_count")

        # FF - Fixation frequency: Total number of ﬁxations divided by the total number of data points for the reading.
        marker_name = "ff_fixation_frequency"
        fixation_frequency = calc_fixation_frequency(fs_left_df, fs_right_df, data_df)
        markers[marker_name] = round(fixation_frequency, 3)
        marker_columns.append(marker_name)
        print("INFO: calculated ff_fixation_frequency")

        # FS - Fixation stability: Mean of distance of movement within a ﬁxation.
        marker_name = "fs_fixation_stability"
        fixation_stability = calc_fixation_stability(data_df)
        markers[marker_name] = round(fixation_stability, 3)
        marker_columns.append(marker_name)
        print("INFO: calculated fs_fixation_stability")

        marker_name = "total_fwd_saccade_count"
        total_fwd_saccade_count = calc_total_saccade_count(fs_left_df, fs_right_df, saccade_direction='F')
        markers[marker_name] = int(total_fwd_saccade_count)
        marker_columns.append(marker_name)
        print("INFO: calculated total_fwd_saccade_count")

        # FSF - Forward saccade frequency: Total number of forward saccades divided by the total number of data points for the reading.
        marker_name = "fsf_fwd_saccade_frequency"
        fwd_saccade_frequency = calc_fwd_saccade_frequency(fs_left_df, fs_right_df, data_df)
        markers[marker_name] = round(fwd_saccade_frequency, 3)
        marker_columns.append(marker_name)
        print("INFO: calculated fsf_fwd_saccade_frequency")

        # SA - Saccadic amplitude: Mean of the speed(distance divided by time taken) of saccades
        marker_name = "sa_saccadic_amplitude"
        saccade_amplitude = calc_saccadic_amplitude(data_df)
        markers[marker_name] = round(saccade_amplitude, 3)
        marker_columns.append(marker_name)
        print("INFO: calculated sa_saccadic_amplitude")

        # MFS - Mean of forward saccade lengths: Summation of the distance of all forward saccades divided by the number of forward saccades
        marker_name = "mfs_mean_fwd_saccade_length"
        fwd_saccade_length = calc_mean_fwd_saccade_length(fs_left_df, fs_right_df)
        markers[marker_name] = round(fwd_saccade_length, 3)
        marker_columns.append(marker_name)
        print("INFO: calculated mfs_mean_fwd_saccade_length")

        marker_name = "total_rev_saccade_count"
        total_rev_saccade_count = calc_total_saccade_count(fs_left_df, fs_right_df, saccade_direction='R')
        markers[marker_name] = int(total_rev_saccade_count)
        marker_columns.append(marker_name)
        print("INFO: calculated total_rev_saccade_count")

        # RF - Regression/Backward saccade frequency: Total number of backward saccades divided by the total number of data points for the reading.
        marker_name = "rf_regression_saccade_frequency"
        reg_saccade_frequency = calc_regression_saccade_frequency(fs_left_df, fs_right_df, data_df)
        markers[marker_name] = round(reg_saccade_frequency, 3)
        marker_columns.append(marker_name)
        print("INFO: calculated rf_regression_saccade_frequency")

        marker_name = "total_time_taken"
        total_time_taken = calc_total_time_taken(data_df)
        markers[marker_name] = total_time_taken
        marker_columns.append(marker_name)
        print("INFO: calculated total_time_taken")

        marker_name = "total_sub_tests"
        total_sub_tests = calc_total_subtest_count(data_df)
        markers[marker_name] = total_sub_tests
        marker_columns.append(marker_name)
        print("INFO: calculated total_sub_tests")

        marker_name = "I-VT Velocity Threshold"
        markers[marker_name] = IVT_VELOCITY_THRESHOLD
        marker_columns.append(marker_name)
        print("INFO: Assigned ivt threshold")

        marker_name = "File Name"
        markers[marker_name] = file
        marker_columns.append(marker_name)
        print("INFO: Assigned file name")

        marker_name = "Sample Count"
        markers[marker_name] = calc_total_data_points(fs_left_df, fs_right_df, data_df)
        marker_columns.append(marker_name)
        print("INFO: Calculated total data points - number of samples in the source file")

        # Store the marker data into a common CSV file.
        store_markers(markers=markers, source_file=file, marker_columns=marker_columns, marker_csv_file=markers_file)
        print("INFO: Stored markers to file")
        print("INFO: Generating graphs")
        #
        # Generate graphs
        #
        l_fix = fs_left_df.query("action=='F'").copy().reset_index(drop=True)
        r_fix = fs_right_df.query("action=='F'").copy().reset_index(drop=True)

        generate_graphs(data, data_df, l_fix, r_fix, fs_left_df, fs_right_df, file)
        print("INFO: Graphs done")
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))

# (FF)Fixation frequency: Total number of ﬁxations divided by the total number of data points for the reading.  - done
# (FSF) Forward saccade frequency: Total number of forward saccades divided by the total number of data points for the reading. - done
# (RF) Regression/Backward saccade frequency: Total number of backward saccades divided by the total number of data points for the reading. - done


# Explanation of various markers:
# (MFL)Mean of ﬁxation lengths: Summationofthelenghtintimeofﬁxations divided by the number of ﬁxations. - done
# (FF)Fixation frequency: Total number of ﬁxations divided by the total number of data points for the reading.  - done
# (FSF) Forward saccade frequency: Total number of forward saccades divided by the total number of data points for the reading. - done
# (FS) Fixation stability: Mean of distance of movement within a ﬁxation. - done
# (SA)Saccadic amplitude: Mean of the speed(distance divided by time taken) of saccades. - done
# (RF) Regression/Backward saccade frequency: Total number of backward saccades divided by the total number of data points for the reading. - done
# (MFS) Mean of forward saccade lengths: Summation of the distance of all forward saccades divided by the number of forward saccades.