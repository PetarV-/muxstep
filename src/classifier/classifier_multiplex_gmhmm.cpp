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

#include <classifier.h>
#include <multiplex_gmhmm.h>

#define DPRINTC(C) printf(#C " = %c\n", (C))
#define DPRINTS(S) printf(#S " = %s\n", (S))
#define DPRINTD(D) printf(#D " = %d\n", (D))
#define DPRINTLLD(LLD) printf(#LLD " = %lld\n", (LLD))
#define DPRINTLF(LF) printf(#LF " = %.5lf\n", (LF))

using namespace std;
typedef unsigned int uint;
typedef long long lld;
typedef unsigned long long llu;

MultiplexGMHMMClassifier::MultiplexGMHMMClassifier(int node_count, int sub_count, int type_count) : node_count(node_count), sub_count(sub_count), type_count(type_count)
{
    positive_model = new MultiplexGMHMM(node_count, sub_count, type_count);
    negative_model = new MultiplexGMHMM(node_count, sub_count, type_count);
    thresholds.clear();
}

MultiplexGMHMMClassifier::MultiplexGMHMMClassifier(int node_count, int sub_count, int type_count, MultiplexGMHMM *positive, MultiplexGMHMM *negative) : node_count(node_count), sub_count(sub_count), type_count(type_count)
{
    positive_model = new MultiplexGMHMM(positive);
    negative_model = new MultiplexGMHMM(negative);
    thresholds.clear();
}

MultiplexGMHMMClassifier::MultiplexGMHMMClassifier(char *filename)
{
    FILE *f = fopen(filename, "r");

    fscanf(f, "%d%d%d", &node_count, &sub_count, &type_count);

    positive_model = new MultiplexGMHMM(node_count, sub_count, type_count, f);
    negative_model = new MultiplexGMHMM(node_count, sub_count, type_count, f);

    fclose(f);
}

MultiplexGMHMMClassifier::~MultiplexGMHMMClassifier()
{
    delete positive_model;
    delete negative_model;
}

Classifier<vector<vector<double> >, bool>* MultiplexGMHMMClassifier::clone()
{
    return new MultiplexGMHMMClassifier(sub_count, type_count, node_count, positive_model, negative_model);
}

void MultiplexGMHMMClassifier::dump(char *filename)
{
    FILE *f = fopen(filename, "w");

    fprintf(f, "%d %d %d\n", node_count, sub_count, type_count);

    positive_model -> dump(f);
    negative_model -> dump(f);

    fclose(f);
}

void MultiplexGMHMMClassifier::dump_muxviz(char *positive_nodes_filename, char *positive_base_layers_filename, char *negative_nodes_filename, char *negative_base_layers_filename)
{
    positive_model -> dump_muxviz_data(positive_nodes_filename, positive_base_layers_filename);
    negative_model -> dump_muxviz_data(negative_nodes_filename, negative_base_layers_filename);
}

void MultiplexGMHMMClassifier::train(vector<pair<vector<vector<double> >, bool> > &training_set)
{
    vector<vector<vector<double> > > train_positive, train_negative;
    int positive_cnt = 0, negative_cnt = 0;
    
    for (uint i=0;i<training_set.size();i++)
    {
        if (training_set[i].second)
        {
            train_positive.push_back(vector<vector<double> >(sub_count));
            train_positive[positive_cnt].resize(sub_count);
            for (int j=0;j<sub_count;j++)
            {
                train_positive[positive_cnt][j].resize(type_count);
                for (int k=0;k<type_count;k++)
                {
                    train_positive[positive_cnt][j][k] = training_set[i].first[j][k];
                }
            }
            positive_cnt++;
        }
        else
        {
            train_negative.push_back(vector<vector<double> >(sub_count));
            train_negative[negative_cnt].resize(sub_count);
            for (int j=0;j<sub_count;j++)
            {
                train_negative[negative_cnt][j].resize(type_count);
                for (int k=0;k<type_count;k++)
                {
                    train_negative[negative_cnt][j][k] = training_set[i].first[j][k];
                }
            }
            negative_cnt++;
        }
    }
    
    positive_model -> train(train_positive);
    negative_model -> train(train_negative);
    
    thresholds.clear();
}

bool MultiplexGMHMMClassifier::classify(vector<vector<double> > &test_data)
{
    double lhood1 = positive_model -> log_likelihood(test_data);
    double lhood0 = negative_model -> log_likelihood(test_data);
    
    thresholds.push_back(lhood1 - lhood0);
    
    return (lhood1 > lhood0);
}

vector<double> MultiplexGMHMMClassifier::get_thresholds()
{
    return thresholds;
}
