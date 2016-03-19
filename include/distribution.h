#ifndef DISTRIBUTION
#define DISTRIBUTION

#include <iostream>
#include <vector>

/*
 This abstract class specifies a probability distribution. Any class extending it can
 be plugged into the main (G)MHMM class, thus allowing for arbitrary output distributions
 being used in muxstep.
*/

class Distribution
{
public:
    virtual ~Distribution() { }
    
    // Create a (deep) copy of the distribution
    virtual Distribution* clone() = 0;
    
    // Train the distribution's parameters on a training set
    virtual void train(std::vector<std::vector<std::pair<int, double> > > &train_set) = 0;
    // Get the probability of producing output x, assuming the sub-output is obs_id
    virtual double get_probability(int obs_id, double x) = 0;
    
    // read/write the distribution to be able to store it for later use
    virtual std::istream& read(std::istream &in) = 0;
    virtual std::ostream& write(std::ostream &out) = 0;
};

#endif
