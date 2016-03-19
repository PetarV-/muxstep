/*
 Petar 'PetarV' Velickovic
 Data Structure: Multiplex GMHMM Classifier
*/

#ifndef CLASSIFIER
#define CLASSIFIER

#include <iostream>
#include <vector>

#include <gmhmm.h>
#include <multiplex_gmhmm.h>
#include <nsga2.h>

/*
 The Multiplex GMHMM Classifier solves a k-class classification problem by constructing k
 Multiplex GMHMMs (one per class) and estimating their parameters using their representative
 sequences in the training set.
 
 Classification of new sequences is then performed by choosing the class corresponding to the
 Multiplex GMHMM that is most likely to have produced it.
 
 Here the special case of k = 2 (binary classification) is focused on in the MultiplexGMHMMClassifier
 class; an experimental MultiplexKClassifier class is provided as a means of extending it to
 the more general case.
*/

// Specifies a generic classifier class that assigns labels to data.
template<typename Data, typename Label>
class Classifier
{
public:
    virtual ~Classifier() { }
    virtual Classifier* clone() = 0; // Needed by the evaluator (for parallel crossvalidation)
    // Train the parameters of the classifier on a given training set
    virtual void train(std::vector<std::pair<Data, Label> > &training_set) = 0;
    // Assign a label to unseen data
    virtual Label classify(Data &test_data) = 0;
    virtual std::vector<double> get_thresholds() = 0; // Needed by the evaluator (for extracting ROC data)
};

// the standard binary classifier
class MultiplexGMHMMClassifier : public Classifier<std::vector<std::pair<int, std::vector<double> > >, bool>
{
private:
    int node_count; // the number of nodes in each layer
    int sub_count;  // the number of sub-outputs
    int type_count; // the number of data types
    MultiplexGMHMM *positive_model;
    MultiplexGMHMM *negative_model;
    
    nsga2_params nsga_p;
    baumwelch_params bw_p;
    
    std::vector<double> thresholds; // Used for generating ROC curve parameters
    
public:
    MultiplexGMHMMClassifier(int node_count, int sub_count, int type_count, nsga2_params nsga_p, baumwelch_params bw_p); // initialise a random multiplex GMHMM classifier
    MultiplexGMHMMClassifier(int node_count, int sub_count, int type_count, nsga2_params nsga_p, baumwelch_params bw_p, MultiplexGMHMM *positive, MultiplexGMHMM *negative); // initialise a multiplex GMHMM classifier from parameters
    ~MultiplexGMHMMClassifier();

    Classifier<std::vector<std::pair<int, std::vector<double> > >, bool>* clone();

    void train(std::vector<std::pair<std::vector<std::pair<int, std::vector<double> > >, bool> > &training_set);
    bool classify(std::vector<std::pair<int, std::vector<double> > > &test_data);
    
    // A special classification method, with a second boolean return value specifying
    // whether the assigned class is chosen with a likelihood margin higher than min_margin
    // (this can be used to warn the user if a classification may be unreliable)
    std::pair<bool, bool> classify_reliable(std::vector<std::pair<int, std::vector<double> > > &test_data, double min_margin);
    
    std::vector<double> get_thresholds();

    // I/O operator overloads
    friend std::istream& operator>>(std::istream &in, MultiplexGMHMMClassifier *&C);
    friend std::ostream& operator<<(std::ostream &out, const MultiplexGMHMMClassifier *C);
};

// Experimental (untested) feature: k-ary classifier
// Most of the implementation is identical in style to above, so repeated comments are omitted
class MultiplexKClassifier : public Classifier<std::vector<std::pair<int, std::vector<double> > >, int>
{
private:
    int node_count;
    int sub_count;
    int type_count;
    int label_count; // how many labels do we have?
    std::vector<MultiplexGMHMM*> models;
    
    nsga2_params nsga_p;
    baumwelch_params bw_p;

public:
    MultiplexKClassifier(int node_count, int sub_count, int type_count, int label_count, nsga2_params nsga_p, baumwelch_params bw_p); // initialise a random k-ary classifier
    MultiplexKClassifier(int node_count, int sub_count, int type_count, int label_count, nsga2_params nsga_p, baumwelch_params bw_p, std::vector<MultiplexGMHMM*> models); // initialise a k-ary classifier from parameters
    ~MultiplexKClassifier();

    Classifier<std::vector<std::pair<int, std::vector<double> > >, int>* clone();

    void train(std::vector<std::pair<std::vector<std::pair<int, std::vector<double> > >, int> > &training_set);
    int classify(std::vector<std::pair<int, std::vector<double> > > &test_data);
    std::pair<int, bool> classify_reliable(std::vector<std::pair<int, std::vector<double> > > &test_data, double min_margin);

    std::vector<double> get_thresholds(); // unnecessary here, returns empty vector

    friend std::istream& operator>>(std::istream &in, MultiplexKClassifier *&C);
    friend std::ostream& operator<<(std::ostream &out, const MultiplexKClassifier *C);
};

#endif
