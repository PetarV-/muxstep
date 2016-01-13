#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <algorithm>
#include <deque>
#include <queue>
#include <stack>
#include <set>
#include <map>
#include <complex>
#include <functional>
#include <chrono>
#include <random>
#include <thread>

#include <classifier.h>
#include <classifier_evaluator.h>

#define DPRINTC(C) printf(#C " = %c\n", (C))
#define DPRINTS(S) printf(#S " = %s\n", (S))
#define DPRINTD(D) printf(#D " = %d\n", (D))
#define DPRINTLLD(LLD) printf(#LLD " = %lld\n", (LLD))
#define DPRINTLF(LF) printf(#LF " = %.5lf\n", (LF))

using namespace std;
typedef unsigned int uint;
typedef long long lld;
typedef unsigned long long llu;

run_result mean_result(vector<run_result> &individual)
{
    int res_cnt = individual.size();
    
    run_result ret;
    ret.accuracy = 0.0;
    ret.precision = 0.0;
    ret.sensitivity = 0.0;
    ret.specificity = 0.0;
    ret.false_positive_rate = 0.0;
    ret.negative_predictive_value = 0.0;
    ret.false_discovery_rate = 0.0;
    ret.mcc = 0.0;
    ret.f1_score = 0.0;
    
    for (int i=0;i<res_cnt;i++)
    {
        ret.accuracy += individual[i].accuracy;
        ret.precision += individual[i].precision;
        ret.sensitivity += individual[i].sensitivity;
        ret.specificity += individual[i].specificity;
        ret.false_positive_rate += individual[i].false_positive_rate;
        ret.negative_predictive_value += individual[i].negative_predictive_value;
        ret.false_discovery_rate += individual[i].false_discovery_rate;
        ret.mcc += individual[i].mcc;
        ret.f1_score += individual[i].f1_score;
    }
    
    ret.accuracy /= (res_cnt * 1.0);
    ret.precision /= (res_cnt * 1.0);
    ret.sensitivity /= (res_cnt * 1.0);
    ret.specificity /= (res_cnt * 1.0);
    ret.false_positive_rate /= (res_cnt * 1.0);
    ret.negative_predictive_value /= (res_cnt * 1.0);
    ret.false_discovery_rate /= (res_cnt * 1.0);
    ret.mcc /= (res_cnt * 1.0);
    ret.f1_score /= (res_cnt * 1.0);
    
    priority_queue<pair<double, pair<int, int> > > pq;
    
    double start_point = individual[0].roc_points[0].first;
    for (int i=0;i<res_cnt;i++)
    {
        if (start_point > individual[i].roc_points[0].first)
        {
            start_point = individual[i].roc_points[0].first;
        }
        
        pq.push(make_pair(individual[i].roc_points[1].first, make_pair(i, 1)));
    }
    
    ret.roc_points.push_back(make_pair(start_point, make_pair(0.0, 0.0)));
    double curr_sum_sensitivity = 0.0, curr_sum_fpr = 0.0;
    
    while (!pq.empty())
    {
        pair<double, pair<int, int> > curr_top = pq.top();
        pq.pop();
        
        double curr_thresh = curr_top.first;
        int curr_node = curr_top.second.first;
        int curr_pos = curr_top.second.second;
        
        double old_sensitivity = individual[curr_node].roc_points[curr_pos - 1].second.first;
        double old_fpr = individual[curr_node].roc_points[curr_pos - 1].second.second;
        
        double new_sensitivity = individual[curr_node].roc_points[curr_pos].second.first;
        double new_fpr = individual[curr_node].roc_points[curr_pos].second.second;
        
        curr_sum_sensitivity += (new_sensitivity - old_sensitivity);
        curr_sum_fpr += (new_fpr - old_fpr);
        
        ret.roc_points.push_back(make_pair(curr_thresh, make_pair(curr_sum_sensitivity / (res_cnt * 1.0), curr_sum_fpr / (res_cnt * 1.0))));
        
        if (uint(curr_pos) < individual[curr_node].roc_points.size() - 1)
        {
            pq.push(make_pair(individual[curr_node].roc_points[curr_pos + 1].first, make_pair(curr_node, curr_pos + 1)));
        }
    }
    
    ret.roc_auc = 0.0;
    for (int i=0;i<res_cnt;i++)
    {
        ret.roc_auc += individual[i].roc_auc;
    }
    ret.roc_auc /= (res_cnt * 1.0);
    
    return ret;
}

run_result stderr_result(vector<run_result> &individual, run_result &means)
{
    int res_cnt = individual.size();
    
    run_result ret;
    ret.accuracy = 0.0;
    ret.precision = 0.0;
    ret.sensitivity = 0.0;
    ret.specificity = 0.0;
    ret.false_positive_rate = 0.0;
    ret.negative_predictive_value = 0.0;
    ret.false_discovery_rate = 0.0;
    ret.mcc = 0.0;
    ret.f1_score = 0.0;
    
    // Must have at least two samples for corrected stddev
    if (res_cnt == 1) return ret;
    
    for (int i=0;i<res_cnt;i++)
    {
        ret.accuracy += (individual[i].accuracy - means.accuracy) * (individual[i].accuracy - means.accuracy);
        ret.precision += (individual[i].precision - means.precision) * (individual[i].precision - means.precision);
        ret.sensitivity += (individual[i].sensitivity - means.sensitivity) * (individual[i].sensitivity - means.sensitivity);
        ret.specificity += (individual[i].specificity - means.specificity) * (individual[i].specificity - means.specificity);
        ret.false_positive_rate += (individual[i].false_positive_rate - means.false_positive_rate) * (individual[i].false_positive_rate - means.false_positive_rate);
        ret.negative_predictive_value += (individual[i].negative_predictive_value - means.negative_predictive_value) * (individual[i].negative_predictive_value - means.negative_predictive_value);
        ret.false_discovery_rate += (individual[i].false_discovery_rate - means.false_discovery_rate) * (individual[i].false_discovery_rate - means.false_discovery_rate);
        ret.mcc += (individual[i].mcc - means.mcc) * (individual[i].mcc - means.mcc);
        ret.f1_score += (individual[i].f1_score - means.f1_score) * (individual[i].f1_score - means.f1_score);
    }
    
    ret.accuracy /= ((res_cnt - 1) * 1.0);
    ret.precision /= ((res_cnt - 1) * 1.0);
    ret.sensitivity /= ((res_cnt - 1) * 1.0);
    ret.specificity /= ((res_cnt - 1) * 1.0);
    ret.false_positive_rate /= ((res_cnt - 1) * 1.0);
    ret.negative_predictive_value /= ((res_cnt - 1) * 1.0);
    ret.false_discovery_rate /= ((res_cnt - 1) * 1.0);
    ret.mcc /= ((res_cnt - 1) * 1.0);
    ret.f1_score /= ((res_cnt - 1) * 1.0);
    
    ret.accuracy = sqrt(ret.accuracy / res_cnt);
    ret.precision = sqrt(ret.precision / res_cnt);
    ret.sensitivity = sqrt(ret.sensitivity / res_cnt);
    ret.specificity = sqrt(ret.specificity / res_cnt);
    ret.false_positive_rate = sqrt(ret.false_positive_rate / res_cnt);
    ret.negative_predictive_value = sqrt(ret.negative_predictive_value / res_cnt);
    ret.false_discovery_rate = sqrt(ret.false_discovery_rate / res_cnt);
    ret.mcc = sqrt(ret.mcc / res_cnt);
    ret.f1_score = sqrt(ret.f1_score / res_cnt);
    
    ret.roc_auc = 0.0;
    for (int i=0;i<res_cnt;i++)
    {
        ret.roc_auc += (individual[i].roc_auc - means.roc_auc) * (individual[i].roc_auc - means.roc_auc);
    }
    ret.roc_auc /= ((res_cnt - 1) * 1.0);
    ret.roc_auc = sqrt(ret.roc_auc / res_cnt);
    
    return ret;
}

void parallel_run(Classifier<vector<vector<double> >, bool> *C, vector<pair<vector<vector<double> >, bool> > &training_set, vector<pair<vector<vector<double> >, bool> > &test_set, run_result &ret)
{
    ret.true_positives = ret.false_positives = 0;
    ret.false_negatives = ret.true_negatives = 0;
    
    C -> train(training_set);
    
    int total = test_set.size();
    int total_positives = 0, total_negatives = 0;
    
    for (uint i=0;i<test_set.size();i++)
    {
        bool expected_inference = test_set[i].second;
        bool inference = C -> classify(test_set[i].first);
        
        if (inference && expected_inference) ret.true_positives++;
        else if (inference && !expected_inference) ret.false_positives++;
        else if (!inference && expected_inference) ret.false_negatives++;
        else if (!inference && !expected_inference) ret.true_negatives++;
        
        if (expected_inference) total_positives++;
        else total_negatives++;
    }
    ret.accuracy = (ret.true_positives + ret.true_negatives) * 1.0 / total * 1.0;
    ret.precision = ret.true_positives * 1.0 / (ret.true_positives + ret.false_positives) * 1.0;
    ret.sensitivity = ret.true_positives * 1.0 / (ret.true_positives + ret.false_negatives) * 1.0;
    ret.specificity = ret.true_negatives * 1.0 / (ret.true_negatives + ret.false_positives) * 1.0;
    ret.false_positive_rate = ret.false_positives * 1.0 / (ret.false_positives + ret.true_negatives) * 1.0;
    ret.negative_predictive_value = ret.true_negatives * 1.0 / (ret.true_negatives + ret.false_negatives) * 1.0;
    ret.false_discovery_rate = ret.false_positives * 1.0 / (ret.false_positives + ret.true_positives) * 1.0;
    
    double S = (ret.true_positives + ret.false_negatives) * 1.0 / total * 1.0;
    double P = (ret.true_positives + ret.false_positives) * 1.0 / total * 1.0;
    
    ret.mcc = (ret.true_positives * 1.0 / total * 1.0 - S * P) / sqrt(P * S * (1 - S) * (1 - P));
    
    ret.f1_score = ret.precision * ret.sensitivity * 2.0 / (ret.precision + ret.sensitivity) * 1.0;
    
    vector<double> thresh = C -> get_thresholds();
    vector<pair<double, bool> > roc_meta;
    roc_meta.resize(test_set.size());
    for (uint i=0;i<test_set.size();i++)
    {
        roc_meta[i] = make_pair(thresh[i], test_set[i].second);
    }
    sort(roc_meta.begin(), roc_meta.end(), greater<pair<double, bool> >());
    
    ret.roc_points.resize(total + 1);
    ret.roc_points[0] = make_pair(roc_meta[0].first + 1.0, make_pair(0.0, 0.0));
    ret.roc_auc = 0.0;
    
    int curr_true_positives = 0, curr_false_positives = 0;
    int curr_false_negatives = total_positives, curr_true_negatives = total_negatives;
    
    for (uint i=0;i<roc_meta.size();i++)
    {
        if (roc_meta[i].second) curr_true_positives++, curr_false_negatives--;
        else curr_false_positives++, curr_true_negatives--;
        
        double old_sensitivity = ret.roc_points[i].second.first;
        double old_fpr = ret.roc_points[i].second.second;
        
        double new_sensitivity = curr_true_positives * 1.0 / (curr_true_positives + curr_false_negatives) * 1.0;
        double new_fpr = curr_false_positives * 1.0 / (curr_false_positives + curr_true_negatives) * 1.0;
        
        if (!roc_meta[i].second) ret.roc_auc += old_sensitivity * (new_fpr - old_fpr);
        ret.roc_points[i+1] = make_pair(roc_meta[i].first, make_pair(new_sensitivity, new_fpr));
    }

    delete C;
}

run_result single_run(Classifier<vector<vector<double> >, bool> *C, vector<pair<vector<vector<double> >, bool> > &training_set, vector<pair<vector<vector<double> >, bool> > &test_set, int num_tests, int num_threads)
{
    run_result max_run;
    max_run.accuracy = -1.0;
    while (num_tests > 0)
    {
        vector<thread> thrs;
        vector<run_result> ret;
        ret.resize(num_threads);
        for (int i=0;i<num_threads - 1;i++)
        {
            thrs.push_back(thread(&parallel_run, C -> clone(), ref(training_set), ref(test_set), ref(ret[i])));
        }
        
        parallel_run(C, training_set, test_set, ret[num_threads - 1]);
        
        for (int i=0;i<num_threads - 1;i++)
        {
            thrs[i].join();
            if (ret[i].accuracy > max_run.accuracy) max_run = ret[i];
        }
        if (ret[num_threads - 1].accuracy > max_run.accuracy) max_run = ret[num_threads - 1];
        
        num_tests -= 1;
    }
    
    return max_run;
}

run_result crossvalidate(Classifier<vector<vector<double> >, bool> *C, vector<pair<vector<vector<double> >, bool> > &training_set, int fold_cnt)
{
    int total = training_set.size();
    int total_positive = 0, total_negative = 0;
    for (uint i=0;i<training_set.size();i++)
    {
        if (training_set[i].second) total_positive++;
        else total_negative++;
    }
    
    int fold_size_positive = total_positive / fold_cnt;
    int fold_size_negative = total_negative / fold_cnt;
    int rem_positive = total_positive % fold_cnt;
    int rem_negative = total_negative % fold_cnt;
    
    vector<vector<pair<vector<vector<double> >, bool> > > folds;
    folds.resize(fold_cnt);
    
    int *fold_size = new int[fold_cnt];
    for (int i=0;i<fold_cnt;i++)
    {
        fold_size[i] = fold_size_positive + (i < rem_positive) + fold_size_negative + (i < rem_negative);
        folds[i].resize(fold_size[i]);
    }
    
    int curr_positive_fold = 0, curr_negative_fold = 0;
    int curr_positive_fold_size = fold_size_positive + (rem_positive > 0);
    int curr_negative_fold_size = fold_size_negative + (rem_negative > 0);
    
    for (uint i=0;i<training_set.size();i++)
    {
        if (training_set[i].second)
        {
            folds[curr_positive_fold][--fold_size[curr_positive_fold]].second = training_set[i].second;
            folds[curr_positive_fold][fold_size[curr_positive_fold]].first.resize(training_set[i].first.size());
            copy(training_set[i].first.begin(), training_set[i].first.end(), folds[curr_positive_fold][fold_size[curr_positive_fold]].first.begin());
            if (--curr_positive_fold_size == 0) curr_positive_fold_size = fold_size_positive + (++curr_positive_fold < rem_positive);
        }
        else
        {
            folds[curr_negative_fold][--fold_size[curr_negative_fold]].second = training_set[i].second;
            folds[curr_negative_fold][fold_size[curr_negative_fold]].first.resize(training_set[i].first.size());
            copy(training_set[i].first.begin(), training_set[i].first.end(), folds[curr_negative_fold][fold_size[curr_negative_fold]].first.begin());
            if (--curr_negative_fold_size == 0) curr_negative_fold_size = fold_size_negative + (++curr_negative_fold < rem_negative);
        }
    }
    
    delete[] fold_size;
    
    vector<run_result> individual;
    individual.resize(fold_cnt);
    for (int i=0;i<fold_cnt;i++)
    {
        vector<pair<vector<vector<double> >, bool> > curr_train, curr_test;
        curr_test = folds[i];
        curr_train.reserve(total - folds[i].size());
        for (int j=0;j<fold_cnt;j++)
        {
            if (i != j) curr_train.insert(curr_train.end(), folds[j].begin(), folds[j].end());
        }
        
        printf("Starting crossvalidation step %d\n", i);
        individual[i] = single_run(C, curr_train, curr_test, 1);
        
        char cur_filename[150];
        sprintf(cur_filename, "results_%02d.out", i);
        dump_result(individual[i], true, cur_filename);
    }
    
    run_result ret = mean_result(individual);
    
    char filename[101];
    sprintf(filename, "results_full.out");
    dump_result(ret, false, filename);
    
    return ret;
}

run_result single_noise_test(Classifier<vector<vector<double> >, bool> *C, vector<pair<vector<vector<double> >, bool> > &training_set, double noise_mean, double noise_stddev, int num_tests)
{
    printf("Performing a noise test, with the following parameters:\n");
    printf("Mean: %.6lf, Standard Deviation: %.6lf, Number of tests: %d\n", noise_mean, noise_stddev, num_tests);
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    normal_distribution<double> noise(noise_mean, noise_stddev);
    
    if (noise_mean == 0.0 && noise_stddev == 0.0) num_tests = 1;
    
    vector<run_result> individual;
    individual.resize(num_tests);
    for (int t=0;t<num_tests;t++)
    {
        vector<pair<vector<vector<double> >, bool> > noisy_training_set;
        noisy_training_set.resize(training_set.size());
        for (uint i=0;i<training_set.size();i++)
        {
            noisy_training_set[i].first.resize(training_set[i].first.size());
            for (uint j=0;j<training_set[i].first.size();j++)
            {
                noisy_training_set[i].first[j].resize(training_set[i].first[j].size());
                for (uint k=0;k<training_set[i].first[j].size();k++)
                {
                    noisy_training_set[i].first[j][k] = training_set[i].first[j][k] + noise(generator);
                }
            }
            noisy_training_set[i].second = training_set[i].second;
        }
        
        printf("Starting noise test #%d\n", t);
        individual[t] = crossvalidate(C, noisy_training_set);
    }
    
    run_result ret = mean_result(individual);
    run_result dev = stderr_result(individual, ret);
    
    char filename[101];
    sprintf(filename, "results_noisy_full.out");
    dump_result(ret, false, filename, noise_mean, noise_stddev);
    sprintf(filename, "results_noisy_stderr.out");
    dump_result(dev, false, filename, noise_mean, noise_stddev);
    
    return ret;
}

void noise_test(Classifier<vector<vector<double> >, bool> *C, vector<pair<vector<vector<double> >, bool> > &training_set, double noise_mean_lo, double noise_mean_step, double noise_mean_hi, double noise_stddev_lo, double noise_stddev_step, double noise_stddev_hi, int num_tests)
{
    printf("Starting a full noise test, with the following parameters:\n");
    printf("Mean: (%.6lf:%.6lf), step size: %.6lf\n", noise_mean_lo, noise_mean_hi, noise_mean_step);
    printf("Standard Deviation: (%.6lf:%.6lf), step size: %.6lf\n", noise_stddev_lo, noise_stddev_hi, noise_stddev_step);
    
    int test_count = 0;
    double mu = noise_mean_lo;
    double sigma = noise_stddev_lo;
    
    do
    {
        do
        {
            char curr_test_fldr[101];
            sprintf(curr_test_fldr, "noisy_test_%d", test_count);
            mkdir(curr_test_fldr, 0775);
            chdir(curr_test_fldr);
            single_noise_test(C, training_set, mu, sigma, num_tests);
            chdir("..");
            sigma += noise_stddev_step;
            test_count++;
        } while (sigma <= noise_stddev_hi);
        
        mu += noise_mean_step;
        sigma = noise_stddev_lo;
    
    } while (mu <= noise_mean_hi);
}

vector<pair<vector<vector<double> >, bool> > extract_data(char* filename)
{
    int total;
    int gene_count, type_count;
    char expected_outcome[101];
    
    vector<pair<vector<vector<double> >, bool> > ret;
    
    FILE *f = fopen(filename, "r");
    
    fscanf(f, "%d", &total);
    fscanf(f, "%d%d", &gene_count, &type_count);
    
    ret.resize(total);
    
    for (int i=0;i<total;i++)
    {
        ret[i].first.resize(gene_count);
        for (int j=0;j<gene_count;j++) ret[i].first[j].resize(type_count);
        
        fscanf(f, "%s", expected_outcome);
        for (int k=0;k<type_count;k++)
        {
            for (int j=0;j<gene_count;j++)
            {
                fscanf(f, "%lf", &ret[i].first[j][k]);
            }
        }
        ret[i].second = (strcmp(expected_outcome, "positive") == 0);
    }
    
    fclose(f);
    
    return ret;
}

void dump_result(run_result &res, bool single_run, char* filename, double noise_mean, double noise_stddev)
{
    FILE *f = fopen(filename, "w");
    
    fprintf(f, "%s results:\n", (single_run ? "Single run" : "Crossvalidation"));
    fprintf(f, "Noise parameters - mean : %.6lf, stddev: %.6lf\n", noise_mean, noise_stddev);
    
    if (single_run)
    {
        fprintf(f, "True positives: %d, False positives: %d\n", res.true_positives, res.false_positives);
        fprintf(f, "False negatives: %d, True negatives: %d\n", res.false_negatives, res.true_negatives);
    }
    
    fprintf(f, "Accuracy: %lf\n", res.accuracy);
    fprintf(f, "Precision: %lf\n", res.precision);
    fprintf(f, "Sensitivity: %lf\n", res.sensitivity);
    fprintf(f, "Specificity: %lf\n", res.specificity);
    fprintf(f, "False positive rate: %lf\n", res.false_positive_rate);
    fprintf(f, "Negative predictive value: %lf\n", res.negative_predictive_value);
    fprintf(f, "False discovery rate: %lf\n", res.false_discovery_rate);
    
    fprintf(f, "Matthews Correlation Coefficient: %lf\n", res.mcc);
    fprintf(f, "F-1 score: %lf\n", res.f1_score);
    
    fprintf(f, "ROC curve parameters (in ascending order):\n");
    for (uint i=0;i<res.roc_points.size();i++)
    {
        fprintf(f, "%lf %lf\n", res.roc_points[i].second.first, res.roc_points[i].second.second);
    }
    
    fprintf(f, "Area under ROC curve: %lf\n", res.roc_auc);
    
    fclose(f);
}
