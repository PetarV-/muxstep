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

MultiplexKClassifier::MultiplexKClassifier(int node_count, int sub_count, int type_count, int label_count) : node_count(node_count), sub_count(sub_count), type_count(type_count), label_count(label_count)
{
    models.resize(label_count);
    for (int i=0;i<label_count;i++)
    {
        models[i] = new MultiplexGMHMM(node_count, sub_count, type_count);
    }
}

MultiplexKClassifier::MultiplexKClassifier(int node_count, int sub_count, int type_count, int label_count, vector<MultiplexGMHMM*> models) : node_count(node_count), sub_count(sub_count), type_count(type_count), label_count(label_count)
{
    this -> models.resize(label_count);
    for (int i=0;i<label_count;i++)
    {
        this -> models[i] = new MultiplexGMHMM(models[i]);
    }
}

MultiplexKClassifier::~MultiplexKClassifier()
{
    for (int i=0;i<label_count;i++)
    {
        delete models[i];
    }
}

Classifier<vector<pair<int, vector<double> > >, int>* MultiplexKClassifier::clone()
{
    return new MultiplexKClassifier(node_count, sub_count, type_count, label_count, models);
}

void MultiplexKClassifier::train(vector<pair<vector<pair<int, vector<double> > >, int> > &training_set)
{
    vector<vector<vector<pair<int, vector<double> > > > > train;
    train.resize(label_count);
    vector<int> cnts;
    cnts.resize(label_count);
    
    for (uint i=0;i<training_set.size();i++)
    {
        int label = training_set[i].second;
        train[label].push_back(vector<pair<int, vector<double> > >());
        train[label][cnts[label]].resize(training_set[i].first.size());
        for (uint j=0;j<training_set[i].first.size();j++)
        {
            train[label][cnts[label]][j].first = training_set[i].first[j].first;
            train[label][cnts[label]][j].second.resize(type_count);
            for (int k=0;k<type_count;k++)
            {
                train[label][cnts[label]][j].second[k] = training_set[i].first[j].second[k];
            }
        }
        cnts[label]++;
    }

    for (int i=0;i<label_count;i++)
    {
        models[i] -> train(train[i]);
    }
}

int MultiplexKClassifier::classify(vector<pair<int, vector<double> > > &test_data)
{
    double max_lhood = models[0] -> log_likelihood(test_data);
    int best = 0;
    
    for (int i=1;i<label_count;i++)
    {
        double curr_lhood = models[i] -> log_likelihood(test_data);
        if (curr_lhood > max_lhood)
        {
            max_lhood = curr_lhood;
            best = i;
        }
    }
    return best;
}

vector<double> MultiplexKClassifier::get_thresholds()
{
    return vector<double>();
}

istream& operator>>(istream& in, MultiplexKClassifier *&C)
{
    if (C != NULL) delete C;
    
    int node_count, sub_count, type_count, label_count;
    in >> node_count >> sub_count >> type_count >> label_count;

    vector<MultiplexGMHMM*> models;
    models.resize(label_count);

    for (int i=0;i<label_count;i++)
    {
        in >> models[i];
    }

    C = new MultiplexKClassifier(node_count, sub_count, type_count, label_count, models);

    return in;
}

ostream& operator<<(ostream &out, const MultiplexKClassifier *C)
{
    out << C -> node_count << " " << C -> sub_count << " " << C -> type_count << " " << C -> label_count << endl;
    
    for (int i=0;i<C -> label_count;i++)
    {
        out << C -> models[i];
    }

    return out;
}

