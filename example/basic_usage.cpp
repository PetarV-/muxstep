#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
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
#include <tuple>
#include <fstream>

#include <classifier.h>
#include <gmhmm.h>
#include <nsga2.h>

#define DPRINTC(C) printf(#C " = %c\n", (C))
#define DPRINTS(S) printf(#S " = %s\n", (S))
#define DPRINTD(D) printf(#D " = %d\n", (D))
#define DPRINTLLD(LLD) printf(#LLD " = %lld\n", (LLD))
#define DPRINTLF(LF) printf(#LF " = %.5lf\n", (LF))

using namespace std;
typedef unsigned int uint;
typedef long long lld;
typedef unsigned long long llu;

// This is a helper method to convert the data given by syn_gen into the format
// expected by the classifier. It also returns the sub-output and type-count for convenience.
tuple<int, int, vector<pair<vector<pair<int, vector<double> > >, bool> > > extract_data(string filename)
{
    int total;
    int sub_count, type_count;
    char expected_outcome[101];
    
    vector<pair<vector<pair<int, vector<double> > >, bool> > ret;
    
    FILE *f = fopen(filename.c_str(), "r");
    
    fscanf(f, "%d", &total);
    fscanf(f, "%d%d", &sub_count, &type_count);
    
    ret.resize(total);
    
    for (int i=0;i<total;i++)
    {
        int curr_size;
        fscanf(f, "%s%d", expected_outcome, &curr_size);
        ret[i].first.resize(curr_size);
        for (int j=0;j<curr_size;j++) 
        {
            fscanf(f, "%d", &ret[i].first[j].first);
            ret[i].first[j].second.resize(type_count);
            for (int k=0;k<type_count;k++)
            {
                fscanf(f, "%lf", &ret[i].first[j].second[k]);
            }
        }
        ret[i].second = (strcmp(expected_outcome, "positive") == 0);
    }
    
    fclose(f);
    
    return make_tuple(sub_count, type_count, ret);
}

// This is a helper method to extract training parameters from a file in a specific format
// This file is expected to have a single key-value pair in each line (see training_params.in for example)
// If a key is unobserved, a default value will be used
pair<nsga2_params, baumwelch_params> extract_parameters(string filename, int type_count)
{
    // Fill up the params with default values
    nsga2_params nsga_p;
    nsga_p.pop_size = 100;
    nsga_p.ft_size = type_count * type_count;
    nsga_p.generations = 250;
    nsga_p.p_crossover = 0.9;
    nsga_p.p_mutation = 1.0 / nsga_p.ft_size;
    nsga_p.di_crossover = 20.0;
    nsga_p.di_mutation = 20.0;
    nsga_p.var_lims.resize(nsga_p.ft_size);
    for (int i=0;i<nsga_p.ft_size;i++)
    {
        nsga_p.var_lims[i] = make_pair(1e-6, 1.0);
    }
    
    baumwelch_params bw_p;
    bw_p.iterations = 10000000;
    bw_p.tolerance = 1e-7;
    
    ifstream f(filename);
    
    string param_key;
    
    // Scan through the file, updating params as we go
    while (f >> param_key)
    {
        if (param_key == "nsga_pop_size") f >> nsga_p.pop_size;
        else if (param_key == "nsga_generations") f >> nsga_p.generations;
        else if (param_key == "nsga_p_crossover") f >> nsga_p.p_crossover;
        else if (param_key == "nsga_p_mutation") f >> nsga_p.p_mutation;
        else if (param_key == "nsga_di_crossover") f >> nsga_p.di_crossover;
        else if (param_key == "nsga_di_mutation") f >> nsga_p.di_mutation;
        else if (param_key == "nsga_var_lims")
        {
            int pos;
            double lo, hi;
            f >> pos >> lo >> hi;
            nsga_p.var_lims[pos] = make_pair(lo, hi);
        }
        else if (param_key == "bw_max_iter") f >> bw_p.iterations;
        else if (param_key == "bw_tolerance") f >> bw_p.tolerance;
    }
    
    f.close();
    
    return make_pair(nsga_p, bw_p);
}

int main()
{
    // Get the data set into a required format
    // (Here using data produced by syn_gen and the provided classifier_evaluator's extract_data method)
    // (This returns a <sub_count, type_count, data> tuple.
    auto data_trn = extract_data("syn_train.out");
    
    // Read the parameters file
    auto params = extract_parameters("training_params.in", get<1>(data_trn));

    // Create a new Multiplex GMHMM Classifier (here, with 4 nodes) using the extracted parameters
    MultiplexGMHMMClassifier *C = new MultiplexGMHMMClassifier(4, get<0>(data_trn), get<1>(data_trn), params.first, params.second);

    // Train the classifier using the extracted data
    C -> train(get<2>(data_trn));
 
    /*
     * Alternatively, a pre-trained model can be input from a file, like so:
     * MultiplexGMHMMClassifier *C;
     * ifstream in("model.txt");
     * in >> C;
     * in.close();
     */

    // Use the trained classifier to make predictions
    // (Here using data produced by syn_gen with the same parameters)
    auto data_tst = get<2>(extract_data("syn_test.out"));
    int total_correct = 0, total = 0;
    for (auto it = data_tst.begin(); it != data_tst.end(); it++)
    {
        // Classify, and report if the margin between the two classes' probabilities is smaller than 0.1
        pair<bool, bool> prediction = C -> classify_reliable(it -> first, 1e-1);
        
        /*
         * If recording reliability of results is not important, can just call the basic classify method:
         * bool prediction = C -> classify(it -> first);
         */
        
        bool expected = it -> second;
        cout << "Predicted class is: " << (prediction.first ? "positive" : "negative") << ". ";
        cout << ((prediction.first == expected) ? "Correct!" : "Incorrect!") << endl;
        cout << "This prediction is " << (prediction.second ? "reliable." : "unreliable!") << endl;
        cout << "----------------------------------------------" << endl;
        total++; 
        if (prediction.first == expected) total_correct++;
    }

    double accuracy = (total_correct * 1.0) / (total * 1.0);
    cout << "Accuracy: " << accuracy << endl;

    // The model could be stored for later use (perhaps into a file)
    ofstream out("model.txt");
    out << C;
    out.close();

    return 0;
}
