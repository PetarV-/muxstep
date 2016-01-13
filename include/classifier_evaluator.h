#ifndef CLASSIFIER_EVAL
#define CLASSIFIER_EVAL

#include <vector>

#include <classifier.h>

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
    
    double mcc;
    double f1_score;
    
    std::vector<std::pair<double, std::pair<double, double> > > roc_points;
    double roc_auc;
};

run_result mean_result(std::vector<run_result> &individual);

run_result stderr_result(std::vector<run_result> &individual, run_result &means);

run_result single_run(Classifier<std::vector<std::vector<double> >, bool> *C, std::vector<std::pair<std::vector<std::vector<double> >, bool> > &training_set, std::vector<std::pair<std::vector<std::vector<double> >, bool> > &test_set, int num_tests = 1, int num_threads = 1);

run_result crossvalidate(Classifier<std::vector<std::vector<double> >, bool> *C, std::vector<std::pair<std::vector<std::vector<double> >, bool> > &training_set, int fold_cnt = 10);

run_result single_noise_test(Classifier<std::vector<std::vector<double> >, bool> *C, std::vector<std::pair<std::vector<std::vector<double> >, bool> > &training_set, double noise_mean, double noise_stddev, int num_tests);

void noise_test(Classifier<std::vector<std::vector<double> >, bool> *C, std::vector<std::pair<std::vector<std::vector<double> >, bool> > &training_set, double noise_mean_lo, double noise_mean_step, double noise_mean_hi, double noise_stddev_lo, double noise_stddev_step, double noise_stddev_hi, int num_tests = 5);

std::vector<std::pair<std::vector<std::vector<double> >, bool> > extract_data(char* filename);

void dump_result(run_result &res, bool single_run, char* filename, double noise_mean = 0.0, double noise_stddev = 0.0);

#endif
