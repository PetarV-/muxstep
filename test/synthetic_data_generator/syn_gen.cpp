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

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: ./syn_gen <input_parameters> <output_file>\n");
        return -1;
    }
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    
    FILE *f = fopen(argv[1], "r");
    FILE *g = fopen(argv[2], "w");
    
    int n, labels, len, types;
    
    fscanf(f, "%d%d%d%d", &n, &labels, &len, &types);
    
    fprintf(g, "%d\n", labels * n);
    fprintf(g, "%d %d\n\n", len, types);
    
    printf("Generating synthetic data...\n");
    
    while (labels--)
    {
        char label[20];
    
        fscanf(f, "%s", label);
    
        normal_distribution<double> **N = new normal_distribution<double>*[types];
        
        for (int t=0;t<types;t++)
        {
            N[t] = new normal_distribution<double>[len];
            for (int i=0;i<len;i++)
            {
                double mean, stddev;
                fscanf(f, "%lf%lf", &mean, &stddev);
                N[t][i] = normal_distribution<double>(mean, stddev);
            }
        }
    
        for (int i=0;i<n;i++)
        {
            fprintf(g, "%s\n", label);
            for (int t=0;t<types;t++)
            {
                for (int j=0;j<len;j++)
                {
                    fprintf(g, "%lf ", N[t][j](generator));
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
