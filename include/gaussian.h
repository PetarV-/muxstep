#ifndef GAUSSIAN
#define GAUSSIAN

#include <iostream>
#include <vector>

#include <distribution.h>

class Gaussian : public Distribution
{
private:
    int obs; // number of sub-outputs
    double *mu, *sigma; // means and standard deviations of each sub-output
    
    // helper function; Phi(x; mean, stddev)
    double gaussian_pdf(double x, double mean, double stdev);
    
public:
    Gaussian(int sub_count);
    Gaussian(int sub_count, double *mu, double *sigma);
    ~Gaussian();
    
    Distribution* clone();
    
    void train(std::vector<std::vector<std::pair<int, double> > > &train_set);
    double get_probability(int obs_id, double x);
    
    std::istream& read(std::istream &in);
    std::ostream& write(std::ostream &out);
};

#endif
