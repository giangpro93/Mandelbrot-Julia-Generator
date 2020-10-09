#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <cstdlib>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "ImageWriter.h"

using namespace std;

struct NameTable
{
	std::string name;
	int value;
};

const char* readSource(const char* fileName);

// A couple simple utility functions:
bool debug = false;
void checkStatus(std::string where, cl_int status, bool abortOnError)
{
	if (debug || (status != 0))
		std::cout << "Step " << where << ", status = " << status << '\n';
	if ((status != 0) && abortOnError)
		exit(1);
}

void reportPlatformInformation(const cl_platform_id& platformIn)
{
	NameTable what[] = {
		{ "CL_PLATFORM_PROFILE:    ", CL_PLATFORM_PROFILE },
		{ "CL_PLATFORM_VERSION:    ", CL_PLATFORM_VERSION },
		{ "CL_PLATFORM_NAME:       ", CL_PLATFORM_NAME },
		{ "CL_PLATFORM_VENDOR:     ", CL_PLATFORM_VENDOR },
		{ "CL_PLATFORM_EXTENSIONS: ", CL_PLATFORM_EXTENSIONS },
		{ "", 0 }
	};
	size_t size;
	char* buf = nullptr;
	int bufLength = 0;
	std::cout << "===============================================\n";
	std::cout << "========== PLATFORM INFORMATION ===============\n";
	std::cout << "===============================================\n";
	for (int i=0 ; what[i].value != 0 ; i++)
	{
		clGetPlatformInfo(platformIn, what[i].value, 0, nullptr, &size);
		if (size > bufLength)
		{
			if (buf != nullptr)
				delete [] buf;
			buf = new char[size];
			bufLength = size;
		}
		clGetPlatformInfo(platformIn, what[i].value, bufLength, buf, &size);
		std::cout << what[i].name << buf << '\n';
	}
	std::cout << "================= END =========================\n\n";
	if (buf != nullptr)
		delete [] buf;
}

void showProgramBuildLog(cl_program pgm, cl_device_id dev)
{
	size_t size;
	clGetProgramBuildInfo(pgm, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
	char* log = new char[size+1];
	clGetProgramBuildInfo(pgm, dev, CL_PROGRAM_BUILD_LOG, size+1, log, nullptr);
	std::cout << "LOG:\n" << log << "\n\n";
	delete [] log;
}

// Typical OpenCL startup
// 1) Platforms
cl_uint numPlatforms = 0;
cl_platform_id* platforms = nullptr;
cl_platform_id curPlatform;
// 2) Devices
cl_uint numDevices = 0;
cl_device_id* devices = nullptr;

// Return value is device index to use; -1 ==> no available devices
int typicalOpenCLProlog(cl_device_type desiredDeviceType)
{
	//-----------------------------------------------------
	// Discover and query the platforms
	//-----------------------------------------------------

	cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
	checkStatus("clGetPlatformIDs-0", status, true);

	platforms = new cl_platform_id[numPlatforms];

	status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
	checkStatus("clGetPlatformIDs-1", status, true);
	int which = 0;
	if (numPlatforms > 1)
	{
		std::cout << "Found " << numPlatforms << " platforms:\n";
		for (int i=0 ; i<numPlatforms ; i++)
		{
			std::cout << i << ": ";
			reportPlatformInformation(platforms[i]);
		}
		which = -1;
		while ((which < 0) || (which >= numPlatforms))
		{
			std::cout << "Which platform do you want to use? ";
			std::cin >> which;
		}
	}
	curPlatform = platforms[which];

	std::cout << "Selected platform: ";
	reportPlatformInformation(curPlatform);

	//----------------------------------------------------------
	// Discover and initialize the devices on a platform
	//----------------------------------------------------------

	status = clGetDeviceIDs(curPlatform, desiredDeviceType, 0, nullptr, &numDevices);
	checkStatus("clGetDeviceIDs-0", status, true);
	if (numDevices <= 0)
	{
		std::cout << "No devices on platform!\n";
		return -1;
	}

	devices = new cl_device_id[numDevices];

	status = clGetDeviceIDs(curPlatform, desiredDeviceType, numDevices, devices, nullptr);
	checkStatus("clGetDeviceIDs-1", status, true);

	// Find a device that supports double precision arithmetic
	int* possibleDevs = new int[numDevices];
	int nPossibleDevs = 0;
	std::cout << "\nLooking for a device that supports double precision...\n";
	for (int idx=0 ; idx<numDevices ; idx++)
	{
		size_t extLength;
		clGetDeviceInfo(devices[idx], CL_DEVICE_EXTENSIONS, 0, nullptr, &extLength);
		char* extString = new char[extLength+1];
		clGetDeviceInfo(devices[idx], CL_DEVICE_EXTENSIONS, extLength+1, extString, nullptr);
		const char* fp64 = strstr(extString, "cl_khr_fp64");
		if (fp64 != nullptr)
			possibleDevs[nPossibleDevs++] = idx;
		delete [] extString;
	}
	if (nPossibleDevs == 0)
	{
		std::cerr << "\nNo device supports double precision.\n";
		return -1;
	}
	size_t nameLength;
	for (int i=0 ; i<nPossibleDevs ; i++)
	{
		clGetDeviceInfo(devices[possibleDevs[i]], CL_DEVICE_NAME, 0, nullptr, &nameLength);
		char* name = new char[nameLength+1];
		clGetDeviceInfo(devices[possibleDevs[i]], CL_DEVICE_NAME, nameLength+1, name, nullptr);
		std::cout << "Device " << i << ": [" << name << "] supports double precision.\n";
		delete [] name;
	}
	if (nPossibleDevs == 1)
	{
		std::cout << "\nNo other device in the requested device category supports double precision.\n"
		          << "You may want to try the -a command line option to see if there are others.\n"
		          << "For now, I will use the one I found.\n";
		return possibleDevs[0];
	}
	int devIndex = -1;
	while ((devIndex < 0) || (devIndex >= nPossibleDevs))
	{
		std::cout << "Which device do you want to use? ";
		std::cin >> devIndex;
	}
	return possibleDevs[devIndex];
}

void process(cl_device_id dev, char option, string inputFilename, string outputFilename) {
    int nRows, nCols, MaxIterations;
    double realMin, realMax, imagMin, imagMax, JuliaRE, JuliaIM, MaxLengthSquared;
    double baseColor[9];
    int nChannels = 3; // RGB

    // Input parameters
    ifstream inFile;
    inFile.open(inputFilename);
    if (inFile.is_open()) {
        inFile>>nRows>>nCols;
        inFile>>MaxIterations;
        inFile>>MaxLengthSquared;
        inFile>>realMin>>realMax;
        inFile>>imagMin>>imagMax;
        inFile>>JuliaRE>>JuliaIM;
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
                inFile>>baseColor[i*3+j];
            }
        }
    }
    else {
        cerr<<"Cannot open file: "<<inputFilename<<endl;
        return;
    }

    // Create image file
    ImageWriter* iw = ImageWriter::create(outputFilename, nCols, nRows, nChannels);
    if (iw == nullptr) {
        exit(1);
    }
    unsigned char* image = new unsigned char[nRows * nCols * nChannels];

    cl_int status;

    // Create the context for a device
    cl_context context = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &status);
    checkStatus("clCreateContext", status, true);

    // Create command queue for a device in the context
    cl_command_queue cmdQueue = clCreateCommandQueue(context, dev, 0, &status);
    checkStatus("clCreateCommandQueue", status, true);

    // Create, compile, and link the program
    const char* programSource[] = { readSource("MandelbrotJuliaGenerator.cl") };
    cl_program program = clCreateProgramWithSource(context, 1, programSource, nullptr, &status);
    checkStatus("clCreateProgramWithSource", status, true);

    status = clBuildProgram(program, 1, &dev, nullptr, nullptr, nullptr);
    if (status != 0)
        showProgramBuildLog(program, dev);
    checkStatus("clBuildProgram", status, true);

    // Configure the work-item structure
    size_t localWorkSize[] = {16, 16};
    size_t globalWorkSize[2];
    globalWorkSize[0] = nCols;
    globalWorkSize[1] = nRows;
    for (int d=0; d<2; d++)
    if (globalWorkSize[d] % localWorkSize[d] != 0) {
        globalWorkSize[d] = ((globalWorkSize[d] / localWorkSize[d]) + 1) * localWorkSize[d];
    }

    cl_kernel kernel;
    size_t imageColorDatasize = nRows*nCols*nChannels*sizeof(double);
    size_t baseColorDatasize = 3*3*sizeof(double);

	// Create content for the image
    double* imageColor = new double[nRows*nCols*nChannels];

    // Create device buffers associated with the context
    cl_mem d_baseColor = clCreateBuffer(context, CL_MEM_READ_ONLY, baseColorDatasize, nullptr, &status);
    checkStatus("clCreateBuffer-baseColor", status, true);
    cl_mem d_imageColor = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageColorDatasize, nullptr, &status);
    checkStatus("clCreateBuffer-imageColor", status, true);

    // Create kernel
    kernel = clCreateKernel(program, "ComputeColor", &status);

    // Set kernel arguments
    int type;
    if (option == 'M') type = 0;
    else type = 1;
    clSetKernelArg(kernel, 0, sizeof(int), &type);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_baseColor);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_imageColor);
    clSetKernelArg(kernel, 3, sizeof(double), &JuliaRE);
    clSetKernelArg(kernel, 4, sizeof(double), &JuliaIM);
    clSetKernelArg(kernel, 5, sizeof(int), &MaxIterations);
    clSetKernelArg(kernel, 6, sizeof(double), &MaxLengthSquared);
    clSetKernelArg(kernel, 7, sizeof(double), &realMin);
    clSetKernelArg(kernel, 8, sizeof(double), &realMax);
    clSetKernelArg(kernel, 9, sizeof(double), &imagMin);
    clSetKernelArg(kernel, 10, sizeof(double), &imagMax);
    clSetKernelArg(kernel, 11, sizeof(int), &nRows);
    clSetKernelArg(kernel, 12, sizeof(int), &nCols);
    clSetKernelArg(kernel, 13, sizeof(int), &nChannels);

    // Write the input to device buffers
    clEnqueueWriteBuffer(cmdQueue, d_baseColor, CL_FALSE, 0, baseColorDatasize, baseColor, 0, nullptr, nullptr);

    // Enqueue the kernel for execution
    clEnqueueNDRangeKernel(cmdQueue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);

    // Read the output buffer back to the host
    status = clEnqueueReadBuffer(cmdQueue, d_imageColor, CL_TRUE, 0, imageColorDatasize, imageColor, 0, nullptr, nullptr);
    checkStatus("clEnqueueReadBuffer", status, true);

    // Release OpenCL kernel and memory object
    clReleaseKernel(kernel);
    clReleaseMemObject(d_baseColor);
    clReleaseMemObject(d_imageColor);

    // Create image
    for (int row=0; row<nRows; row++) {
        for (int col=0; col<nCols; col++) {
            for (int channel=0; channel<nChannels; channel++) {
                int location = row*nCols*nChannels + col*nChannels + channel;
                image[location] = static_cast<unsigned char>(imageColor[location]*255.0 + 0.5);
            }
        }
    }

    iw->writeImage(image);
    iw->closeImageFile();

    // Free heap allocated arrays
    delete[] imageColor;

    // Free OpenCL resources
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseContext(context);

    // Free host resources
	delete [] platforms;
	delete [] devices;

    // Free writeImage resources
    delete iw;
    delete [] image;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr<<"Usage: "<<argv[0]<<" M/J params.txt imageFileOut.png"<<endl;
    }
    else {
        char option = argv[1][0];
        string inputFilename = argv[2];
        string outputFilename = argv[3];

        cl_device_type devType = CL_DEVICE_TYPE_DEFAULT;
        int devIndex = typicalOpenCLProlog(devType);
        if (devIndex >= 0) {
            process(devices[devIndex], option, inputFilename, outputFilename);
        }
    }
}
