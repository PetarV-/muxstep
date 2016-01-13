#ifndef CLASSIFIER
#define CLASSIFIER

#include <vector>

#include <gmhmm.h>
#include <multiplex_gmhmm.h>

template<typename Data, typename Label>
class Classifier
{
public:
    virtual ~Classifier() { }
    virtual Classifier* clone() = 0; // Needed by the evaluator (for parallel crossvalidation)
    virtual void train(std::vector<std::pair<Data, Label> > &training_set) = 0;
    virtual Label classify(Data &test_data) = 0;
    virtual std::vector<double> get_thresholds() = 0; // Needed by the evaluator (for extracting ROC data)
};

class MultiplexGMHMMClassifier : public Classifier<std::vector<std::vector<double> >, bool>
{
private:
    int sub_count;  // the number of sub-outputs
    int type_count; // the number of data types
    int node_count; // the number of nodes in each layer
    MultiplexGMHMM* positive_model;
    MultiplexGMHMM* negative_model;
    
    std::vector<double> thresholds;
    
public:
    MultiplexGMHMMClassifier(int sub_count, int type_count, int node_count = 4); // initialise a random multiplex GMHMM
    MultiplexGMHMMClassifier(int sub_count, int type_count, int node_count, MultiplexGMHMM* positive, MultiplexGMHMM* negative); // initialise a multiplex GMHMM from parameters 
    MultiplexGMHMMClassifier(const char* filename); // load from a file given in the necessary format
    ~MultiplexGMHMMClassifier();

    Classifier* clone();
    
    void dump(const char* filename); // dump the model parameters into a file for later use
    void dump_muxviz(char *positive_nodes_filename, char *positive_base_layers_filename, char *negative_nodes_filename, char *negative_base_layers_filename); // dump the positive and negative models into a format readable by muxViz

    void train(std::vector<std::pair<std::vector<std::vector<double> >, bool> > &training_set);
    bool classify(std::vector<std::vector<double> > &test_data);
    
    std::vector<double> get_thresholds();
};

#endif
