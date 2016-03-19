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
#include <gmhmm.h>
#include <multiplex_gmhmm.h>
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

MultiplexGMHMMClassifier::MultiplexGMHMMClassifier(int node_count, int sub_count, int type_count, nsga2_params nsga_p, baumwelch_params bw_p) : node_count(node_count), sub_count(sub_count), type_count(type_count), nsga_p(nsga_p), bw_p(bw_p)
{
    positive_model = new MultiplexGMHMM(node_count, sub_count, type_count);
    negative_model = new MultiplexGMHMM(node_count, sub_count, type_count);
    thresholds.clear();
}

MultiplexGMHMMClassifier::MultiplexGMHMMClassifier(int node_count, int sub_count, int type_count, nsga2_params nsga_p, baumwelch_params bw_p, MultiplexGMHMM *positive, MultiplexGMHMM *negative) : node_count(node_count), sub_count(sub_count), type_count(type_count), nsga_p(nsga_p), bw_p(bw_p)
{
    positive_model = new MultiplexGMHMM(positive);
    negative_model = new MultiplexGMHMM(negative);
    thresholds.clear();
}

MultiplexGMHMMClassifier::~MultiplexGMHMMClassifier()
{
    delete positive_model;
    delete negative_model;
}

Classifier<vector<pair<int, vector<double> > >, bool>* MultiplexGMHMMClassifier::clone()
{
    return new MultiplexGMHMMClassifier(node_count, sub_count, type_count, nsga_p, bw_p, positive_model, negative_model);
}

void MultiplexGMHMMClassifier::train(vector<pair<vector<pair<int, vector<double> > >, bool> > &training_set)
{
    vector<vector<pair<int, vector<double> > > > train_positive, train_negative;
    int positive_cnt = 0, negative_cnt = 0;
    
    for (uint i=0;i<training_set.size();i++)
    {
        if (training_set[i].second)
        {
            train_positive.push_back(vector<pair<int, vector<double> > >());
            train_positive[positive_cnt].resize(training_set[i].first.size());
            for (uint j=0;j<training_set[i].first.size();j++)
            {
                train_positive[positive_cnt][j].first = training_set[i].first[j].first;
                train_positive[positive_cnt][j].second.resize(type_count);
                for (int k=0;k<type_count;k++)
                {
                    train_positive[positive_cnt][j].second[k] = training_set[i].first[j].second[k];
                }
            }
            positive_cnt++;
        }
        else
        {
            train_negative.push_back(vector<pair<int, vector<double> > >());
            train_negative[negative_cnt].resize(training_set[i].first.size());
            for (uint j=0;j<training_set[i].first.size();j++)
            {
                train_negative[negative_cnt][j].first = training_set[i].first[j].first;
                train_negative[negative_cnt][j].second.resize(type_count);
                for (int k=0;k<type_count;k++)
                {
                    train_negative[negative_cnt][j].second[k] = training_set[i].first[j].second[k];
                }
            }
            negative_cnt++;
        }
    }
    
    positive_model -> train(train_positive, nsga_p, bw_p);
    negative_model -> train(train_negative, nsga_p, bw_p);
    
    thresholds.clear();
}

bool MultiplexGMHMMClassifier::classify(vector<pair<int, vector<double> > > &test_data)
{
    double lhood1 = positive_model -> log_likelihood(test_data);
    double lhood0 = negative_model -> log_likelihood(test_data);
    
    thresholds.push_back(lhood1 - lhood0);
    
    return (lhood1 > lhood0);
}

pair<bool, bool> MultiplexGMHMMClassifier::classify_reliable(vector<pair<int, vector<double> > > &test_data, double min_margin)
{
    double lhood1 = positive_model -> log_likelihood(test_data);
    double lhood0 = negative_model -> log_likelihood(test_data);
    
    return make_pair((lhood1 > lhood0), (fabs(lhood1 - lhood0) > min_margin));
}

vector<double> MultiplexGMHMMClassifier::get_thresholds()
{
    return thresholds;
}

istream& operator>>(istream& in, MultiplexGMHMMClassifier *&C)
{
    if (C != NULL) delete C;
    
    int node_count, sub_count, type_count;
    in >> node_count >> sub_count >> type_count;

    nsga2_params nsga_p;
    in >> nsga_p;
    
    baumwelch_params bw_p;
    in >> bw_p;
    
    MultiplexGMHMM *positive_model;
    MultiplexGMHMM *negative_model;
    in >> positive_model;
    in >> negative_model;

    C = new MultiplexGMHMMClassifier(node_count, sub_count, type_count, nsga_p, bw_p, positive_model, negative_model);

    return in;
}

ostream& operator<<(ostream &out, const MultiplexGMHMMClassifier *C)
{
    out << C -> node_count << " " << C -> sub_count << " " << C -> type_count << endl;
    
    out << C -> nsga_p;
    out << C -> bw_p;
    
    out << C -> positive_model;
    out << C -> negative_model;

    return out;
}

