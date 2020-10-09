# uncomment the following lines if running on linux
CF_EXTRA = -Wno-deprecated-declarations
OPENCL_LIB = -lOpenCL
#uncomment the following lines if running on Mac OS
# CF_EXTRA = -DCL_SILENCE_DEPRECATION
# OPENCL_LIB = -framework OpenCL

PROG = MandelbrotJuliaGenerator

CPP = g++ -fPIC -std=c++11
LINK = g++ -fPIC

# The following will be needed when compiling:
INCLUDES = -I../Packed3DArray -I../ImageWriter

# The following will be needed when linking:
LIBRARIES = ../lib/libCOGLImageWriter.so

$(PROG): $(PROG).o readSource.o $(LIBRARIES)
	$(LINK) -g -o $(PROG) $(PROG).o readSource.o $(LIBRARIES) $(OPENCL_LIB)

../lib/libCOGLImageWriter.so: ../ImageWriter/ImageWriter.h ../ImageWriter/ImageWriter.c++
	(cd ../ImageWriter; make)

$(PROG).o: $(PROG).c++
	$(CPP) -g -c $(CF_EXTRA) $(INCLUDES) $(PROG).c++

readSource.o: readSource.c++
	$(CPP) -g -c $(CF_EXTRA) readSource.c++

clean:
	rm *.o
