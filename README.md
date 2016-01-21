# muxstep

TODO...

## Building

You will need clang++ to compile the library (on Mac OS X, this compiler is installed by default).

Simply run

> $ make

while in the root directory of the repository, and the libraries will be made in the lib/ folder.

## Usage example

The folder example/ contains basic_usage.cpp, a file that demonstrates a basic way in which the muxstep library might be used.

If you would like to compile it using the static library, run the following:

> $ clang++ -std=c++11 -I../include basic_usage.cpp -L../lib -lmuxstep -o basic

If you would like to use the dynamic library instead, run the following:

> $ clang++ -std=c++11 -I../include basic_usage.cpp -L../lib -lmuxstep.dyn -o basic
> $LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/muxstep/lib/
> $export LD_LIBRARY_PATH

where /path/to/muxstep refers to the (absolute) path to the root repository folder.

After doing this, you may run the example using

> $ basic

(Note, it takes some time for training to complete!)

## License

MIT
