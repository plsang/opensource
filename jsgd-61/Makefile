MEX=/usr/local/matlab2009b/bin/mex


# common flags
CFLAGS=-g -Wall -fPIC -O3


all: mex octave py

# A good BLAS implementation is required for jsgd. When compiling for
# matlab, we use the implementation shipped with it (-lmwblas, that
# also requires 64-bit integers). Else (for Octave and Python) we use
# the system library. On Linux (Debian 6 and Fedore 14), the Atlas
# implementation is reasonably fast, but must be pointed to explicitly
# because -lblas is not efficient:

BLASLIB=/net/per900a/raid0/plsang/usr.local/lib/libblas.so


# matlab-specific version of x_matrix.o
matlab/x_matrix.o: c/x_matrix.c c/x_matrix.h
	gcc -c -o $@ $< $(CFLAGS) -Dinteger=long -DBLAS_WITH_THREADS -fopenmp

# We assume that the system BLAS uses 32-bit ints and is not
# multithreaded. Octave and Python version of x_matrix.o:
c/x_matrix.o: c/x_matrix.c c/x_matrix.h
	gcc -c -o $@ $< $(CFLAGS) -Dinteger=int  -fopenmp

c/jsgd.o: c/jsgd.c c/jsgd.h c/x_matrix.h
	gcc -c -o $@ $< $(CFLAGS) -fopenmp

mex: c/jsgd.o matlab/x_matrix.o matlab/jsgd_train.c
	(cd matlab && $(MEX) -g jsgd_train.c ../c/jsgd.o x_matrix.o -lmwblas LDFLAGS="\$$LDFLAGS -fopenmp" )

# there does not seem to be a way to add flags to the mkoctfile
# command line (that compiles octave mex files). Therefore we call gcc
# explicitly. The options below can be found with
#    touch e.c ; mkoctfile --mex -v e.c

OCTAVEOPTIONS=-I/usr/include/octave-3.6.4 -I/usr/include/octave-3.6.4/octave -fexceptions -fstack-protector --param=ssp-buffer-size=4  -L/usr/lib64/octave/3.6.4 -loctinterp -loctave  -Wl,-Bsymbolic

octave: matlab/jsgd_train.mex

matlab/jsgd_train.mex: matlab/jsgd_train.c c/jsgd.o c/x_matrix.o 
	gcc -shared -o $@ $^ $(CFLAGS) $(OCTAVEOPTIONS) -fopenmp

# the Python wrapper requires SWIG (works with 2.0.1), and the
# developer packages for Python and numpy. From these includes,
# Python.h and numpy/arrayobject.h should be accessible:

PYTHONINCLUDES=-I/net/per900a/raid0/plsang/usr.local/include/python2.7 -I/net/per900a/raid0/plsang/usr.local/lib/python2.7/site-packages/numpy/core/include

# also produces python/jsgd_wrap.py
python/jsgd_wrap_wrap.c: python/jsgd_wrap.swg c/jsgd.h c/x_matrix.h
	(cd python && swig -python -D__restrict__=  jsgd_wrap.swg ) 


py: python/_jsgd_wrap.so

python/_jsgd_wrap.so: python/jsgd_wrap_wrap.c c/jsgd.o c/x_matrix.o 
	gcc -shared -o $@ $^ $(CFLAGS) -fopenmp $(PYTHONINCLUDES)  $(BLASLIB)

clean:
	rm -f c/*.o python/*.o matlab/*.o matlab/*.mex* python/jsgd_wrap_wrap.c python/jsgd_wrap.py

