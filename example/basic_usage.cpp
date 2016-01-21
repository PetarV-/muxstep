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
#include <fstream>

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

int main()
{
    // Get the data set into a required format
    // (Here using data produced by syn_gen and the provided classifier_evaluator's extract_data method)
    // (This returns a <sub_count, type_count, data> tuple.
    auto data_trn = extract_data("syn_train.out");

    // Create a new GMHMM (here, with 4 nodes)
    MultiplexGMHMMClassifier *C = new MultiplexGMHMMClassifier(4, get<0>(data_trn), get<1>(data_trn));

    // Train the classifier using the data
    C -> train(get<2>(data_trn));
 
    /*
     * Alternatively, a pre-trained model can be input from a file, like so:
     * MultiplexGMHMMClassifier *C;
     * ifstream in("model.txt");
     * in >> C;
     * in.close();
     */

    // Use the trained classifier to make predictions
    // (Here using data produced by syn_gen with the same parameters)
    auto data_tst = get<2>(extract_data("syn_test.out"));
    int total_correct = 0, total = 0;
    for (auto it = data_tst.begin(); it != data_tst.end(); it++)
    {
        bool prediction = C -> classify(it -> first);
        bool expected = it -> second;
        cout << "Predicted class is: " << (prediction ? "positive" : "negative") << ". ";
        cout << ((prediction == expected) ? "Correct!" : "Incorrect!") << endl;
        total++; 
        if (prediction == expected) total_correct++;
    }

    double accuracy = (total_correct * 1.0) / (total * 1.0);
    cout << "Accuracy: " << accuracy << endl;

    // The model could be stored for later use (perhaps into a file)
    ofstream out("model.txt");
    out << C;
    out.close();

    return 0;
}
