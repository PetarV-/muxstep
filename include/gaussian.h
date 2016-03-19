#ifndef GAUSSIAN
#define GAUSSIAN

#include <iostream>
#include <vector>

#include <distribution.h>

/*
 The Gaussian class represents a Gaussian (normal) distribution, which is
 at the same time a basic example on how to extend the Distribution class, 
 and also the default distribution used by this model in previous studies.
*/

class Gaussian : public Distribution
{
private:
    int obs; // number of sub-outputs
    double *mu, *sigma; // means and standard deviations of each sub-output
    
    // helper function; Phi(x; mean, stddev)
    double gaussian_pdf(double x, double mean, double stdev);
    
public:
    // create a new untrained Gaussian distribution with a known number of sub-outputs
    Gaussian(int sub_count);
    // copy a known Gaussian distribution from its parameters
    Gaussian(int sub_count, double *mu, double *sigma);
    ~Gaussian();
    
    // Overridden methods from Distribution
    
    Distribution* clone();
    
    void train(std::vector<std::vector<std::pair<int, double> > > &train_set);
    double get_probability(int obs_id, double x);
    
    std::istream& read(std::istream &in);
    std::ostream& write(std::ostream &out);
};

#endif
