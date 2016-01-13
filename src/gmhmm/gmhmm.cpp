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
#include <tuple>
#include <chrono>
#include <random>

#include <gmhmm.h>

#define DPRINTC(C) printf(#C " = %c\n", (C))
#define DPRINTS(S) printf(#S " = %s\n", (S))
#define DPRINTD(D) printf(#D " = %d\n", (D))
#define DPRINTLLD(LLD) printf(#LLD " = %lld\n", (LLD))
#define DPRINTLF(LF) printf(#LF " = %.5lf\n", (LF))

using namespace std;
typedef unsigned int uint;
typedef long long lld;
typedef unsigned long long llu;

default_random_engine gen;
uniform_real_distribution<double> rnd_real(0.0, 1.0);

GMHMM::GMHMM(int n, int obs) : n(n), obs(obs)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    gen = default_random_engine(seed);
    
    double total = 0.0;
    this -> pi = new double[n];
    for (int i=0;i<n;i++)
    {
        this -> pi[i] = rnd_real(gen);
        total += this -> pi[i];
    }
    for (int i=0;i<n;i++)
    {
        this -> pi[i] /= total;
    }
    
    this -> T = new double*[n];
    for (int i=0;i<n;i++)
    {
        this -> T[i] = new double[n];
        total = 0.0;
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] = rnd_real(gen);
            total += this -> T[i][j];
        }
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] /= total;
        }
    }
    
    this -> O = new double*[n];
    for (int i=0;i<n;i++)
    {
        this -> O[i] = new double[obs];
        total = 0.0;
        for (int j=0;j<obs;j++)
        {
            this -> O[i][j] = rnd_real(gen);
            total += this -> O[i][j];
        }
        for (int j=0;j<n;j++)
        {
            this -> O[i][j] /= total;
        }
    }
    
    this -> mu = new double[obs];
    this -> sigma = new double[obs];
}

GMHMM::GMHMM(int n, int obs, double *pi, double **T, double **O, double *mu, double *sigma) : n(n), obs(obs)
{
    this -> pi = new double[n];
    for (int i=0;i<n;i++) this -> pi[i] = pi[i];
    
    this -> T = new double*[n];
    for (int i=0;i<n;i++)
    {
        this -> T[i] = new double[n];
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] = T[i][j];
        }
    }
    
    this -> O = new double*[n];
    for (int i=0;i<n;i++)
    {
        this -> O[i] = new double[obs];
        for (int j=0;j<obs;j++)
        {
            this -> O[i][j] = O[i][j];
        }
    }
    
    this -> mu = new double[obs];
    this -> sigma = new double[obs];
    for (int i=0;i<obs;i++)
    {
        this -> mu[i] = mu[i];
        this -> sigma[i] = sigma[i];
    }
}

GMHMM::~GMHMM()
{
    delete[] pi;
    
    for (int i=0;i<n;i++)
    {
        delete[] T[i];
        delete[] O[i];
    }
    delete[] T;
    delete[] O;
    
    delete[] mu;
    delete[] sigma;
}

tuple<double**, double*, double> GMHMM::forward(vector<pair<double, int> > &Y)
{
    int Ti = Y.size();
    
    double **alpha = new double*[Ti];
    for (int i=0;i<Ti;i++)
    {
        alpha[i] = new double[n];
    }
    double *c = new double[Ti];
    
    double sum = 0.0;
    for (int i=0;i<n;i++)
    {
        alpha[0][i] = pi[i] * O[i][Y[0].second] * get_probability(Y[0].second, Y[0].first);
        sum += alpha[0][i];
    }
    c[0] = 1.0 / sum;
    for (int i=0;i<n;i++)
    {
        alpha[0][i] /= sum;
    }
    
    for (int t=1;t<Ti;t++)
    {
        sum = 0.0;
        for (int i=0;i<n;i++)
        {
            alpha[t][i] = 0.0;
            for (int j=0;j<n;j++)
            {
                alpha[t][i] += alpha[t-1][j] * T[j][i];
            }
            alpha[t][i] *= O[i][Y[t].second] * get_probability(Y[t].second, Y[t].first);
            sum += alpha[t][i];
        }
        
        c[t] = 1.0 / sum;
        for (int i=0;i<n;i++)
        {
            alpha[t][i] /= sum;
        }
    }
    
    double log_L = 0.0;
    for (int i=0;i<Ti;i++) log_L -= log(c[i]);
    
    return make_tuple(alpha, c, log_L);
}

double** GMHMM::backward(vector<pair<double, int> > &Y, double *c)
{
    int Ti = Y.size();
    
    double **beta = new double*[Ti];
    for (int i=0;i<Ti;i++)
    {
        beta[i] = new double[n];
    }
    for (int i=0;i<n;i++) beta[Ti-1][i] = 1.0;
    
    for (int t=Ti-2;t>=0;t--)
    {
        for (int i=0;i<n;i++)
        {
            beta[t][i] = 0.0;
            for (int j=0;j<n;j++)
            {
                beta[t][i] += T[i][j] * O[j][Y[t+1].second] * get_probability(Y[t+1].second, Y[t+1].first) * beta[t+1][j];
            }
            beta[t][i] *= c[t+1];
        }
    }
    
    return beta;
}

void GMHMM::baumwelch(vector<vector<double> > &Ys, int iterations, double tolerance)
{
    vector<vector<pair<double, int> > > sorted_Ys;
    sorted_Ys.resize(Ys.size());
    
    for (uint l=0;l<Ys.size();l++)
    {
        sorted_Ys[l].resize(Ys[l].size());
        for (uint i=0;i<Ys[l].size();i++) sorted_Ys[l][i] = make_pair(Ys[l][i], i);
        sort(sorted_Ys[l].begin(), sorted_Ys[l].end());
    }
    
    double ***alpha = new double**[sorted_Ys.size()];
    double ***beta = new double**[sorted_Ys.size()];
    double **c = new double*[sorted_Ys.size()];
    
    double PP, QQ;
    
    double lhood = 0.0;
    double oldlhood = 0.0;
    
    for (int iter=0;iter<iterations;iter++)
    {
        lhood = 0.0;
        
        for (uint l=0;l<sorted_Ys.size();l++)
        {
            tuple<double**, double*, double> x = forward(sorted_Ys[l]);
            alpha[l] = get<0>(x);
            c[l] = get<1>(x);
            lhood += get<2>(x);
            beta[l] = backward(sorted_Ys[l], c[l]);
        }
        
        double **nextO = new double*[n];
        for (int i=0;i<n;i++) nextO[i] = new double[obs];
        
        for (int i=0;i<n;i++)
        {
            pi[i] = 0.0;
            for (uint l=0;l<sorted_Ys.size();l++)
            {
                pi[i] += alpha[l][0][i] * beta[l][0][i];
            }
            pi[i] /= sorted_Ys.size();
            
            QQ = 0.0;
            
            for (int k=0;k<obs;k++)
            {
                nextO[i][k] = 0.0;
            }
            
            for (uint l=0;l<sorted_Ys.size();l++)
            {
                for (uint t=0;t<sorted_Ys[l].size()-1;t++)
                {
                    double curr = alpha[l][t][i] * beta[l][t][i];
                    QQ += curr;
                    nextO[i][sorted_Ys[l][t].second] += curr;
                }
            }
            
            for (int j=0;j<n;j++)
            {
                PP = 0.0;
                for (uint l=0;l<sorted_Ys.size();l++)
                {
                    for (uint t=0;t<sorted_Ys[l].size()-1;t++)
                    {
                        PP += alpha[l][t][i] * O[j][sorted_Ys[l][t+1].second] * get_probability(sorted_Ys[l][t+1].second, sorted_Ys[l][t+1].first) * beta[l][t+1][j] * c[l][t+1];
                    }
                }
                T[i][j] *= PP / QQ;
            }
            
            for (uint l=0;l<sorted_Ys.size();l++)
            {
                int lim = sorted_Ys[l].size() - 1;
                double curr = alpha[l][lim][i] * beta[l][lim][i];
                QQ += curr;
                nextO[i][sorted_Ys[l][lim].second] += curr;
            }
            
            for (int k=0;k<obs;k++)
            {
                nextO[i][k] /= QQ;
            }
        }
        
        for (uint l=0;l<sorted_Ys.size();l++)
        {
            for (uint t=0;t<sorted_Ys[l].size();t++)
            {
                delete[] alpha[l][t];
                delete[] beta[l][t];
            }
            delete[] alpha[l];
            delete[] beta[l];
            delete[] c[l];
        }
        
        for (int i=0;i<n;i++)
        {
            delete[] O[i];
        }
        delete[] O;
        
        O = nextO;
        
        printf("%.10lf\n", lhood - oldlhood);
        
        if (fabs(lhood - oldlhood) < tolerance)
        {
            printf("Converged after %d iterations!\n", iter + 1);
            break;
        }
        
        if ((iter + 1) % 20 == 0)
        {
            printf("Completed %d iterations\n", iter + 1);
        }
        
        oldlhood = lhood;
    }
    
    printf("Baum-Welch procedure completed.\n");
    
    printf("START-STATE VECTOR:\n");
    for (int i=0;i<n;i++) printf("%lf ", pi[i]);
    printf("\n");
    
    printf("TRANSITION MATRIX:\n");
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<n;j++)
        {
            printf("%lf ", T[i][j]);
        }
        printf("\n");
    }
    
    printf("OBSERVATION MATRIX\n");
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<obs;j++)
        {
            printf("%lf ", O[i][j]);
        }
        printf("\n");
    }
}

void GMHMM::baumwelch(vector<vector<pair<double, int> > > &sorted_Ys, int iterations, double tolerance)
{
    double ***alpha = new double**[sorted_Ys.size()];
    double ***beta = new double**[sorted_Ys.size()];
    double **c = new double*[sorted_Ys.size()];
    
    double PP, QQ;
    
    double lhood = 0.0;
    double oldlhood = 0.0;
    
    for (int iter=0;iter<iterations;iter++)
    {
        lhood = 0.0;
        
        for (uint l=0;l<sorted_Ys.size();l++)
        {
            tuple<double**, double*, double> x = forward(sorted_Ys[l]);
            alpha[l] = get<0>(x);
            c[l] = get<1>(x);
            lhood += get<2>(x);
            beta[l] = backward(sorted_Ys[l], c[l]);
        }
        
        double **nextO = new double*[n];
        for (int i=0;i<n;i++) nextO[i] = new double[obs];
        
        for (int i=0;i<n;i++)
        {
            pi[i] = 0.0;
            for (uint l=0;l<sorted_Ys.size();l++)
            {
                pi[i] += alpha[l][0][i] * beta[l][0][i];
            }
            pi[i] /= sorted_Ys.size();
            
            QQ = 0.0;
            
            for (int k=0;k<obs;k++)
            {
                nextO[i][k] = 0.0;
            }
            
            for (uint l=0;l<sorted_Ys.size();l++)
            {
                for (uint t=0;t<sorted_Ys[l].size()-1;t++)
                {
                    double curr = alpha[l][t][i] * beta[l][t][i];
                    QQ += curr;
                    nextO[i][sorted_Ys[l][t].second] += curr;
                }
            }
            
            for (int j=0;j<n;j++)
            {
                PP = 0.0;
                for (uint l=0;l<sorted_Ys.size();l++)
                {
                    for (uint t=0;t<sorted_Ys[l].size()-1;t++)
                    {
                        PP += alpha[l][t][i] * O[j][sorted_Ys[l][t+1].second] * get_probability(sorted_Ys[l][t+1].second, sorted_Ys[l][t+1].first) * beta[l][t+1][j] * c[l][t+1];
                    }
                }
                T[i][j] *= PP / QQ;
            }
            
            for (uint l=0;l<sorted_Ys.size();l++)
            {
                int lim = sorted_Ys[l].size() - 1;
                double curr = alpha[l][lim][i] * beta[l][lim][i];
                QQ += curr;
                nextO[i][sorted_Ys[l][lim].second] += curr;
            }
            
            for (int k=0;k<obs;k++)
            {
                nextO[i][k] /= QQ;
            }
        }
        
        for (uint l=0;l<sorted_Ys.size();l++)
        {
            for (uint t=0;t<sorted_Ys[l].size();t++)
            {
                delete[] alpha[l][t];
                delete[] beta[l][t];
            }
            delete[] alpha[l];
            delete[] beta[l];
            delete[] c[l];
        }
        
        for (int i=0;i<n;i++)
        {
            delete[] O[i];
        }
        delete[] O;
        
        O = nextO;
        
        printf("%.10lf\n", lhood - oldlhood);
        
        if (fabs(lhood - oldlhood) < tolerance)
        {
            printf("Converged after %d iterations!\n", iter + 1);
            break;
        }
        
        if ((iter + 1) % 20 == 0)
        {
            printf("Completed %d iterations\n", iter + 1);
        }
        
        oldlhood = lhood;
    }
    
    printf("Baum-Welch procedure completed.\n");
    
    printf("START-STATE VECTOR:\n");
    for (int i=0;i<n;i++) printf("%lf ", pi[i]);
    printf("\n");
    
    printf("TRANSITION MATRIX:\n");
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<n;j++)
        {
            printf("%lf ", T[i][j]);
        }
        printf("\n");
    }
    
    printf("OBSERVATION MATRIX\n");
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<obs;j++)
        {
            printf("%lf ", O[i][j]);
        }
        printf("\n");
    }
}

double GMHMM::get_pi(int x)
{
    return this -> pi[x];
}

double GMHMM::get_T(int i, int j)
{
    return this -> T[i][j];
}

double GMHMM::get_O(int x, int y)
{
    return this -> O[x][y];
}

double GMHMM::get_probability(int obs_id, double x)
{
    return gaussian_pdf(x, mu[obs_id], sigma[obs_id]);
}

void GMHMM::train(vector<vector<double> > &train_set)
{
    // get means and std. deviations
    for (int i=0;i<obs;i++)
    {
        mu[i] = 0.0;
        sigma[i] = 0.0;
    }
    
    for (uint i=0;i<train_set.size();i++)
    {
        for (int j=0;j<obs;j++)
        {
            mu[j] += train_set[i][j];
        }
    }
    
    for (int i=0;i<obs;i++) mu[i] /= train_set.size();
    
    for (uint i=0;i<train_set.size();i++)
    {
        for (int j=0;j<obs;j++)
        {
            sigma[j] += (train_set[i][j] - mu[j]) * (train_set[i][j] - mu[j]);
        }
    }
    
    for (int i=0;i<obs;i++) sigma[i] = sqrt(sigma[i] / (train_set.size() - 1));
    
    printf("Mus/Sigmas calculated: \n");
    for (int i=0;i<obs;i++)
    {
        printf("(%lf, %lf)\n", mu[i], sigma[i]);
    }
    
    // reset the initial probabilities
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    gen = default_random_engine(seed);
    
    double total = 0.0;
    for (int i=0;i<n;i++)
    {
        this -> pi[i] = rnd_real(gen);
        total += this -> pi[i];
    }
    for (int i=0;i<n;i++)
    {
        this -> pi[i] /= total;
    }
    
    for (int i=0;i<n;i++)
    {
        total = 0.0;
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] = rnd_real(gen);
            total += this -> T[i][j];
        }
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] /= total;
        }
    }
    
    for (int i=0;i<n;i++)
    {
        total = 0.0;
        for (int j=0;j<obs;j++)
        {
            this -> O[i][j] = rnd_real(gen);
            total += this -> O[i][j];
        }
        for (int j=0;j<n;j++)
        {
            this -> O[i][j] /= total;
        }
    }
    
    // now run the Baum-Welch algorithm
    baumwelch(train_set, 10000000, 1e-7);
}

void GMHMM::train(vector<vector<pair<double, int> > > &train_set)
{
    // get means and std. deviations
    for (int i=0;i<obs;i++)
    {
        mu[i] = 0.0;
        sigma[i] = 0.0;
    }
    
    for (uint i=0;i<train_set.size();i++)
    {
        for (int j=0;j<obs;j++)
        {
            mu[train_set[i][j].second] += train_set[i][j].first;
        }
    }
    
    for (int i=0;i<obs;i++) mu[i] /= train_set.size();
    
    for (uint i=0;i<train_set.size();i++)
    {
        for (int j=0;j<obs;j++)
        {
            sigma[train_set[i][j].second] += (train_set[i][j].first - mu[train_set[i][j].second]) * (train_set[i][j].first - mu[train_set[i][j].second]);
        }
    }
    
    for (int i=0;i<obs;i++) sigma[i] = sqrt(sigma[i] / (train_set.size() - 1));
    
    printf("Mus/Sigmas calculated: \n");
    for (int i=0;i<obs;i++)
    {
        printf("(%lf, %lf)\n", mu[i], sigma[i]);
    }
    
    // reset the initial probabilities
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    gen = default_random_engine(seed);
    
    double total = 0.0;
    for (int i=0;i<n;i++)
    {
        this -> pi[i] = rnd_real(gen);
        total += this -> pi[i];
    }
    for (int i=0;i<n;i++)
    {
        this -> pi[i] /= total;
    }
    
    for (int i=0;i<n;i++)
    {
        total = 0.0;
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] = rnd_real(gen);
            total += this -> T[i][j];
        }
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] /= total;
        }
    }
    
    for (int i=0;i<n;i++)
    {
        total = 0.0;
        for (int j=0;j<obs;j++)
        {
            this -> O[i][j] = rnd_real(gen);
            total += this -> O[i][j];
        }
        for (int j=0;j<n;j++)
        {
            this -> O[i][j] /= total;
        }
    }
    
    // now run the Baum-Welch algorithm
    baumwelch(train_set, 10000000, 1e-7);
}

double GMHMM::log_likelihood(vector<double> &test_data)
{
    vector<pair<double, int> > sorted_data;
    sorted_data.resize(test_data.size());
    for (uint i=0;i<test_data.size();i++) sorted_data[i] = make_pair(test_data[i], i);
    sort(sorted_data.begin(), sorted_data.end());
    
    tuple<double**, double*, double> x = forward(sorted_data);
    double **alpha = get<0>(x);
    double *c = get<1>(x);
    
    for (uint i=0;i<test_data.size();i++)
    {
        delete[] alpha[i];
    }
    delete[] alpha;
    delete[] c;
    
    return get<2>(x);
}

double GMHMM::log_likelihood(vector<pair<double, int> > &sorted_data)
{
    tuple<double**, double*, double> x = forward(sorted_data);
    double **alpha = get<0>(x);
    double *c = get<1>(x);
    
    for (uint i=0;i<sorted_data.size();i++)
    {
        delete[] alpha[i];
    }
    delete[] alpha;
    delete[] c;
    
    return get<2>(x);
}
