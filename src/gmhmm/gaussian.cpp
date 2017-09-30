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

#include <distribution.h>
#include <gaussian.h>

using namespace std;

#define DPRINTC(C) printf(#C " = %c\n", (C))
#define DPRINTS(S) printf(#S " = %s\n", (S))
#define DPRINTD(D) printf(#D " = %d\n", (D))
#define DPRINTLLD(LLD) printf(#LLD " = %lld\n", (LLD))
#define DPRINTLF(LF) printf(#LF " = %.5lf\n", (LF))

// helper function; Phi(x; mean, stddev)
double Gaussian::gaussian_pdf(double x, double mean, double stdev)
{
    // simply compute exp(-(x - mu)^2/2*sigma^2)/sqrt(2*pi*sigma^2)
    double E = x - mean;
    E *= -E;
    E /= 2 * stdev * stdev;
    double ret = exp(E);
    return ret / (stdev * sqrt(2 * M_PI));
}

// create a new untrained Gaussian distribution with a known number of sub-outputs
Gaussian::Gaussian(int sub_count) : obs(sub_count)
{
    mu = new double[sub_count];
    sigma = new double[sub_count];
}

// copy a known Gaussian distribution from its parameters
Gaussian::Gaussian(int sub_count, double *mu, double *sigma) : obs(sub_count)
{
    this -> mu = new double[obs];
    this -> sigma = new double[obs];
    for (int i=0;i<obs;i++)
    {
        this -> mu[i] = mu[i];
        this -> sigma[i] = sigma[i];
    }
}
    
Gaussian::~Gaussian()
{
    delete[] mu;
    delete[] sigma;
}

// Create a (deep) copy of the distribution
Distribution* Gaussian::clone()
{
    return new Gaussian(obs, mu, sigma);
}

// Train the distribution's parameters on a training set
void Gaussian::train(vector<vector<pair<int, double> > > &train_set)
{
    int *cnt = new int[obs];
    
    for (int i=0;i<obs;i++)
    {
        mu[i] = 0.0;
        sigma[i] = 0.0;
        cnt[i] = 0;
    }
    
    // compute sample means and std. deviations for each sub-output individually
    for (uint i=0;i<train_set.size();i++)
    {
        for (uint j=0;j<train_set[i].size();j++)
        {
            mu[train_set[i][j].first] += train_set[i][j].second;
            cnt[train_set[i][j].first]++;
        }
    }
    
    for (int i=0;i<obs;i++) mu[i] /= cnt[i];
    
    for (uint i=0;i<train_set.size();i++)
    {
        for (uint j=0;j<train_set[i].size();j++)
        {
            sigma[train_set[i][j].first] += (train_set[i][j].second - mu[train_set[i][j].first]) * (train_set[i][j].second - mu[train_set[i][j].first]);
        }
    }
        
    for (int i=0;i<obs;i++) sigma[i] = sqrt(sigma[i] / (cnt[i] - 1));
        
    delete[] cnt;
}

// Get the probability of producing output x, assuming the sub-output is obs_id
double Gaussian::get_probability(int obs_id, double x)
{
    // simply return the already defined probability density function
    return gaussian_pdf(x, mu[obs_id], sigma[obs_id]);
}

// read the distribution from an input stream
istream& Gaussian::read(istream &in)
{
    in >> this -> obs;
    this -> mu = new double[obs];
    this -> sigma = new double[obs];
    for (int i=0;i<obs;i++)
    {
        in >> mu[i] >> sigma[i];
    }
    
    return in;
}

// write the distribution to an output stream
ostream& Gaussian::write(ostream &out)
{
    out << obs << endl;
        
    for (int i=0;i<obs;i++)
    {
        out << mu[i] << " " << sigma[i] << endl;
    }
    
    return out;
}


