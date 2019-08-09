import ntpath
from os import listdir
from os.path import isfile, join
import glob
import numpy as np
import pandas as pd

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


# %matplotlib inline


class PupilIVT(object):
    def __init__(self, sample_freq=40, threshold_velocity=30):
        self.sample_freq = sample_freq
        self.threshold_velocity = threshold_velocity

    #
    # Process horizontal eye movement
    # Calculate velocity at each sample point and mark whether it's a fixation or not.
    #
    def calc_horizontal_velocity(self, data_df):
        velocity_list_left = [0]
        fixation_list_left = [0]
        velocity_list_right = [0]
        fixation_list_right = [0]
        distance_list_left = [0]  #absolute distance travelled
        distance_list_right = [0] #absolute distance travelled
        net_distance_list_left = [0]  # distance travelled - with sign.
        net_distance_list_right = [0] # distance travelled - with sign.

        prev_pos_left = data_df.loc[0, "LeftPupil.X"]
        prev_pos_right = data_df.loc[0, "RightPupil.X"]
        prev_time = data_df.loc[0, "TimeMillis"]

        for idx, row in data_df.loc[1:, :].iterrows():
            cur_pos_left = row["LeftPupil.X"]
            cur_pos_right = row["RightPupil.X"]
            cur_time = row["TimeMillis"]

            cur_velocity_left = abs(cur_pos_left - prev_pos_left) / abs(cur_time - prev_time)
            cur_velocity_right = abs(cur_pos_right - prev_pos_right) / abs(cur_time - prev_time)

            cur_distance_left = abs(cur_pos_left - prev_pos_left)
            cur_distance_right = abs(cur_pos_right - prev_pos_right)

            net_distance_left  = cur_pos_left - prev_pos_left
            net_distance_right = cur_pos_right - prev_pos_right

            velocity_list_left.append(cur_velocity_left)
            velocity_list_right.append(cur_velocity_right)

            distance_list_left.append(cur_distance_left)  # added
            distance_list_right.append(cur_distance_right)  # added

            net_distance_list_left.append(net_distance_left)
            net_distance_list_right.append(net_distance_right)

            # If velocity at the sample point is lesser than threshold velocity, then it's a fixation. mark 1
            # otherwise, mark 0
            if (abs(cur_velocity_left) < self.threshold_velocity):
                fixation_list_left.append(1)
            else:
                fixation_list_left.append(0)

            if (abs(cur_velocity_right) < self.threshold_velocity):
                fixation_list_right.append(1)
            else:
                fixation_list_right.append(0)

            prev_pos_left = cur_pos_left
            prev_pos_right = cur_pos_right
            prev_time = cur_time

        data_df["velocity_left"] = pd.Series(velocity_list_left)  # velocity in this reading
        data_df["velocity_right"] = pd.Series(velocity_list_right)

        data_df["distance_left"] = pd.Series(distance_list_left)  # distance moved in this reading (absolute value)
        data_df["distance_right"] = pd.Series(distance_list_right)

        data_df["net_distance_left"]  = pd.Series(net_distance_list_left) # distance moved in this reading (relative value, with sign)
        data_df["net_distance_right"] = pd.Series(net_distance_list_right)

        data_df["fixation_left"] = pd.Series(fixation_list_left)  # flag=1: this reading is a fixation. flag=0: saccade
        data_df["fixation_right"] = pd.Series(fixation_list_right)
        return data_df


    # Traverse through each sample, merge together fixations and saccades.
    # returns a dataframe containing fixation start,end row numbers, duration and number of samples per fixation/saccade.
    # action=S (saccade), F (fixation)
    def process_fixation_saccades(self, data_df, tag):
        fix_sac_list = []

        if(tag == "left"):
            fixation_column = "fixation_left"
            distance_column = "distance_left"
        else:
            fixation_column = "fixation_right"
            distance_column = "distance_right"

        fix_count = 0
        sac_count = 0
        fix_time = 0
        sac_time = 0

        prev_slno = 0
        prev_fix = data_df.loc[0, fixation_column]
        prev_time = data_df.loc[0, "TimeMillis"]
        begin_time = prev_time
        begin_slno = 0
        distance_travelled = 0

        if (prev_fix == 1):
            fix_count = 1
            fix_time = prev_time
        else:
            sac_count = 1
            sac_time = prev_time

        for idx, row in data_df.loc[1:, :].iterrows():
            cur_fix = row[fixation_column]
            cur_time = row["TimeMillis"]
            cur_slno = row["slno"]
            cur_distance = row[distance_column]

            if (prev_fix == cur_fix):
                if (cur_fix == 1):
                    fix_count = fix_count + 1
                    fix_time = cur_time
                else:
                    sac_count = sac_count + 1
                    sac_time = cur_time

                distance_travelled = distance_travelled + cur_distance
            else:
                if (prev_fix == 1):
                    fix_time = fix_time - begin_time
                    # print("Row " + str(idx) + " Fixation ends. Duration: " + str(fix_time) + " Count: " + str(fix_count) )
                    fix_sac = {"action": "F", "duration": fix_time, "distance": distance_travelled, "samples": fix_count, "start_id": begin_slno,
                               "end_id": prev_slno}
                    sac_count = 1
                    sac_time = cur_time
                    begin_time = prev_time
                    begin_slno = cur_slno
                else:
                    sac_time = sac_time - begin_time
                    # print("Row " + str(idx) + " Saccad ends. Duration: " + str(sac_time) + " Count: " + str(sac_count) )
                    fix_sac = {"action": "S", "duration": sac_time, "distance": distance_travelled, "samples": sac_count, "start_id": begin_slno,
                               "end_id": prev_slno}
                    fix_count = 1
                    fix_time = cur_time
                    begin_time = prev_time
                    begin_slno = cur_slno

                fix_sac_list.append(fix_sac)
                distance_travelled = cur_distance

            prev_fix = cur_fix
            prev_time = cur_time
            prev_slno = cur_slno

        fs_df = pd.DataFrame(fix_sac_list)
        return fs_df

    def calc_saccade_direction(self, data_df, fs_df, tag="left"):
        saccade_direction_list = []
        target_column_name = "saccade_direction"

        if(tag == "left"):
            data_x_column_name = "LeftPupil.X"
        else:
            data_x_column_name = "RightPupil.X"

        for idx, row in fs_df.iterrows():

            if (row["action"] == 'F'):  # Skip fixations
                saccade_direction_list.append("-")
                continue

            start_idx = row["start_id"]
            end_idx = row["end_id"]

            start_x = data_df.loc[start_idx, data_x_column_name]
            end_x = data_df.loc[end_idx, data_x_column_name]
            # print("IDX:{0} START:{1} END:{2}".format(idx,start_x, end_x))
            # If the X position at the beginning of a saccade is greater than the X position at the end of a saccade,
            # then we treat this as a reverse saccade.
            if (end_x < start_x):
                saccade_direction_list.append("R")
            else:
                saccade_direction_list.append("F")

        saccade_direction_sr = pd.Series(saccade_direction_list)

        fs_df[target_column_name] = saccade_direction_sr
        return fs_df