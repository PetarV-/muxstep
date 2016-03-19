#ifndef CLASSIFIER_EVAL
#define CLASSIFIER_EVAL

#include <tuple>
#include <vector>

#include <classifier.h>
#include <gmhmm.h>
#include <nsga2.h>

/*
 This file defines all of the subroutines used by the evaluation suite.
 
 The evaluation suite is capable of performing stratified k-fold crossvalidation (typically, k = 10)
 over a given data set with known labels and then computing means of a variety of metrics.
 
 It is also capable of performing noise testing, which will first apply Gaussian noise to 
 the data set before doing the crossvalidation; the metrics are averaged for each noise 
 level over five runs.
*/

// A data structure containing metrics of a single testing/crossvalidation run
struct run_result
{
    int true_positives, false_positives;
    int false_negatives, true_negatives;
    
    double accuracy;
    double precision;
    double sensitivity;
    double specificity;
    double false_positive_rate;
    double negative_predictive_value;
    double false_discovery_rate;
    
    double mcc; // Matthews Correlation Coefficient
    double f1_score; // Harmonic mean of precision and recall
    
    std::vector<std::pair<double, std::pair<double, double> > > roc_points; // Points on the ROC curve
    double roc_auc; // Area under the ROC curve
};

// Averages a sequence of metrics
run_result mean_result(std::vector<run_result> &individual);

// Computes the standard error in the averages of a sequence of metrics
run_result stderr_result(std::vector<run_result> &individual, run_result &means);

// Perform a single evaluation run, for a particular training and testing set split.
run_result single_run(Classifier<std::vector<std::pair<int, std::vector<double> > >, bool> *C, std::vector<std::pair<std::vector<std::pair<int, std::vector<double> > >, bool> > &training_set, std::vector<std::pair<std::vector<std::pair<int, std::vector<double> > >, bool> > &test_set, int num_tests = 1, int num_threads = 1);

// Perform a single stratified k-fold crossvalidation run
run_result crossvalidate(Classifier<std::vector<std::pair<int, std::vector<double> > >, bool> *C, std::vector<std::pair<std::vector<std::pair<int, std::vector<double> > >, bool> > &training_set, int num_tests = 1, int num_threads = 1, int fold_cnt = 10);

// Perform a single noise test (for particular noise parameters)
run_result single_noise_test(Classifier<std::vector<std::pair<int, std::vector<double> > >, bool> *C, std::vector<std::pair<std::vector<std::pair<int, std::vector<double> > >, bool> > &training_set, double noise_mean, double noise_stddev, int num_tests);

// Perform a full noise test
void noise_test(Classifier<std::vector<std::pair<int, std::vector<double> > >, bool> *C, std::vector<std::pair<std::vector<std::pair<int, std::vector<double> > >, bool> > &training_set, double noise_mean_lo, double noise_mean_step, double noise_mean_hi, double noise_stddev_lo, double noise_stddev_step, double noise_stddev_hi, int num_tests = 5);

// A helper subroutine for extracting the data set from a syn_gen-formatted file
std::tuple<int, int, std::vector<std::pair<std::vector<std::pair<int, std::vector<double> > >, bool> > > extract_data(std::string filename);

// A helper subroutine for extracting the training algorithm parameters (from a file in the expected format)
std::pair<nsga2_params, baumwelch_params> extract_parameters(std::string filename, int type_count);

// Dump the results into a given file
void dump_result(run_result &res, bool single_run, char* filename, double noise_mean = 0.0, double noise_stddev = 0.0);

#endif
