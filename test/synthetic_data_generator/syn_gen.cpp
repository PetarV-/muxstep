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
#include <chrono>
#include <random>

#define DPRINTC(C) printf(#C " = %c\n", (C))
#define DPRINTS(S) printf(#S " = %s\n", (S))
#define DPRINTD(D) printf(#D " = %d\n", (D))
#define DPRINTLLD(LLD) printf(#LLD " = %lld\n", (LLD))
#define DPRINTLF(LF) printf(#LF " = %.5lf\n", (LF))

using namespace std;
typedef long long lld;
typedef unsigned long long llu;

bool compare_norm(pair<int, vector<double> > a, pair<int, vector<double> > b)
{
    double norm_sq_a = 0.0, norm_sq_b = 0.0;

    assert(a.second.size() == b.second.size());

    for (uint i=0;i<a.second.size();i++)
    {
        norm_sq_a += a.second[i] * a.second[i];
        norm_sq_b += b.second[i] * b.second[i];
    }

    return norm_sq_a > norm_sq_b;
}

int main(int argc, char **argv)
{
    if (argc != 4 || (argv[3][0] != 'Y' && argv[3][0] != 'N'))
    {
        printf("Usage: ./syn_gen <input_parameters> <output_file> <sort? [Y/N]>\n");
        return -1;
    }
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    
    FILE *f = fopen(argv[1], "r");
    FILE *g = fopen(argv[2], "w");
    bool to_sort = (argv[3][0] == 'Y');
    
    int n, labels, sub, types;
    int lo, hi;

    fscanf(f, "%d%d%d%d", &n, &labels, &sub, &types);
    fscanf(f, "%d%d", &lo, &hi);
    
    uniform_int_distribution<int> length_distribution(lo, hi);

    fprintf(g, "%d\n", labels * n);
    fprintf(g, "%d %d\n\n", sub, types);
    
    printf("Generating synthetic data...\n");
    
    while (labels--)
    {
        char label[20];
    
        fscanf(f, "%s", label);
    
        normal_distribution<double> **N = new normal_distribution<double>*[types];
        
        for (int t=0;t<types;t++)
        {
            N[t] = new normal_distribution<double>[sub];
            for (int i=0;i<sub;i++)
            {
                double mean, stddev;
                fscanf(f, "%lf%lf", &mean, &stddev);
                N[t][i] = normal_distribution<double>(mean, stddev);
            }
        }

        for (int i=0;i<n;i++)
        {
            int len = length_distribution(generator);
            fprintf(g, "%s %d\n", label, len);

            vector<pair<int, vector<double> > > data;
            data.resize(len);
            
            for (int j=0;j<len;j++)
            {
                data[j].first = j;
                data[j].second.resize(types);
                for (int t=0;t<types;t++)
                {
                    data[j].second[t] = N[t][j](generator);
                }
            }

            if (to_sort) sort(data.begin(), data.end(), compare_norm);

            for (int j=0;j<len;j++)
            {
                fprintf(g, "%d ", data[j].first);
                for (int t=0;t<types;t++)
                {
                    fprintf(g, "%lf ", data[j].second[t]);
                }
                fprintf(g, "\n");
            }
            fprintf(g, "\n");
        }
        
        for (int t=0;t<types;t++) delete[] N[t];
        delete[] N;
    }
    
    fclose(f);
    fclose(g);
    
    printf("Done. Synthetic data written to %s.\n", argv[2]);
    
    return 0;
}
