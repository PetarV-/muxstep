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

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 5)
    {
        printf("Usage: ./simple_tester <node_count> <data_set_file> (<noise_mean_lo>:<noise_mean_step>:<noise_mean_hi> <noise_stddev_lo>:<noise_stddev_step>:<noise_stddev_hi>)\n");
        return -1;
    }
    
    int node_count;
    sscanf(argv[1], "%d", &node_count);
    tuple<int, int, vector<pair<vector<pair<int, vector<double> > >, bool> > > data = extract_data(argv[2]);

    Classifier<vector<pair<int, vector<double> > >, bool> *C = new MultiplexGMHMMClassifier(node_count, get<0>(data), get<1>(data));
    
    if (argc == 5) 
    {
        double mu_lo, mu_step, mu_hi;
        double sigma_lo, sigma_step, sigma_hi;
        sscanf(argv[3], "%lf:%lf:%lf", &mu_lo, &mu_step, &mu_hi);
        sscanf(argv[4], "%lf:%lf:%lf", &sigma_lo, &sigma_step, &sigma_hi);
        noise_test(C, get<2>(data), mu_lo, mu_step, mu_hi, sigma_lo, sigma_step, sigma_hi);
    }
    else
    {
        crossvalidate(C, get<2>(data)); 
    }
   
    return 0;
}
