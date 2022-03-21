"""
Created by Dániel Varga;
"""

from descriptiveness import *
import datetime as dt
import time
import json
import os

def main():
    f = open("test_settings.json", "r")
    settings = json.load(f)

    print("-------------------------------- START! --------------------------------")
    start_t = time.time()

    # Create the evaluator object
    evaluator = DescriptivenessEvaluator(settings)

    # Run the evaluation and get the results
    auc_pairs, auc_pairs_2, recMean, precMean = evaluator.get_auc_pairs()

    end_t = time.time()

    if not os.path.exists("logs/"):
        os.makedirs("logs/")
    if not os.path.exists("logs_image/"):
        os.makedirs("logs_image/")
    if not os.path.exists("results/"):
        os.makedirs("results/")

    auc_result_file_name = "results/" + settings["descriptor"] + "_auc.csv"
    prc_result_file_name = "results/" + settings["descriptor"] + "_prc.csv"
    np.savetxt(auc_result_file_name, auc_pairs_2)
    np.savetxt(prc_result_file_name, np.array([precMean, recMean]))


    tt = dt.datetime.now()
    curr_time = tt.strftime("%m_%d_%H_%M_%S")
    logfile_name = "logs/"+settings["descriptor"]+"_log_" + curr_time + ".log"
    logfile = open(logfile_name, "w")

    logfile.write("Parameters:\n")
    logfile.write('-- sampleNum:                ' + str(settings["sampleNum"])               + '\n')
    logfile.write('-- keypoint_support_radius:  ' + str(settings["keypoint_support_radius"]) + '\n')
    logfile.write('-- repeat:                   ' + str(settings["repeat"])                  + '\n')
    logfile.write('-- cloud_pairs:              ' + str(settings["cloud_pairs"])             + '\n')
    logfile.write('-- dim:                      ' + str(settings["dim"])                     + '\n')
    logfile.write('-- descriptor:               "' + str(settings["descriptor"])             + '"\n')
    logfile.write('-- dataset_name:             "' + str(settings["dataset_name"])           + '"\n')
    logfile.write('-- dataset_path:             "' + str(settings["dataset_path"])           + '"\n')
    logfile.write('-- input_cloud_format:       "' + str(settings["input_cloud_format"])     + '"\n')
    logfile.write('-- persistence_file:         "' + str(settings["persistence_file"])       + '"\n')
    logfile.write('-- is_binary:                ' + str(settings["is_binary"])               + '\n')
    logfile.write('-- read_persisted_data:      ' + str(settings["read_persisted_data"])     + '\n')
    logfile.write('-- use_custom_metric:        ' + str(settings["use_custom_metric"])       + '\n')
    logfile.write('-- required time(sec):       ' + str(end_t-start_t)                       + '\n')
    logfile.write('-- bits: '                     + str(settings["bits"])                    + '\n')

    logfile.write("\nResults:")
    logfile.write("\n-- mean auc_pairs:\n")
    logfile.write(str(np.mean(auc_pairs)))
    logfile.write("\n-- std auc_pairs:\n")
    for auc_pair_2 in auc_pairs_2:
        logfile.write(str(np.std(auc_pair_2)))
        logfile.write(", ")
    logfile.write("\n\n-- auc_pairs:\n")
    logfile.write(str(auc_pairs))
    logfile.write("\n-- auc_pairs_2:\n")
    logfile.write(str(auc_pairs_2))
    logfile.write("\n\n")
    logfile.write("\n-- Recall Means:\n")
    logfile.write(str(recMean))
    logfile.write("\n")
    logfile.write("\n-- Precision Means:\n")
    logfile.write(str(precMean))
    logfile.write("\n")
    logfile.close() 

    save_name = "logs_image/"+settings["descriptor"]+"_log_" + curr_time + ".png"
    plot_prc(recMean, precMean, save_name, False)

    print("--------------------------------- END! ---------------------------------")


if __name__ == "__main__":
    main()
