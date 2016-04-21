# muxstep

muxstep is an open-source C++ multiplex HMM library for making inferences on multiple data types.

## Building

You will need `clang++` to compile the library (on Mac OS X, this compiler is installed by default).

Simply run

    $ make

while in the root directory of the repository, and the libraries (both static and dynamic) will be created in the `lib/` folder. This command will also compile the documentation files (supplementary data as given in the original publication) in `doc/muxstep-suppl.pdf` (`pdflatex` is required for compiling the documentation).

## Usage example

The folder `example/` contains `basic_usage.cpp`, a file that demonstrates a basic way in which the muxstep library might be used.

If you would like to compile it using the static library, run the following:

    $ clang++ -std=c++11 -I../include basic_usage.cpp -L../lib -lmuxstep -o basic

If you would like to use the dynamic library instead, run the following:

    $ clang++ -std=c++11 -I../include basic_usage.cpp -L../lib -lmuxstep.dyn -o basic
    $ LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/muxstep/lib/
    $ export LD_LIBRARY_PATH

where `/path/to/muxstep` refers to the (absolute) path to the root repository folder.

After doing this, you may run the example using

    $ ./basic

(Note, it takes some time for training to complete!)

## License

MIT

## References

If you make advantage of muxstep or derive it within your research, please cite the following article:

Veličković, P. and Liò, P. (2016) [muxstep: an open-source C++ multiplex HMM library for making inferences on multiple data types.](http://bioinformatics.oxfordjournals.org/content/early/2016/04/13/bioinformatics.btw196) *Bioinformatics*

The models described here were originally investigated in the following manuscript:

Veličković, P. and Liò, P. (2015) [Molecular multiplex network inference using Gaussian mixture hidden Markov models.](http://comnet.oxfordjournals.org/content/early/2015/12/25/comnet.cnv029) *Journal of Complex Networks*
