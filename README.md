# optixGcodeViewer
Simple G-code viewer using OptiX 7

Tested on Windows 10 with:
 * Visual Studio Community 2017
 * CUDA 10.1 and CUDA 11.5
 * OptiX 7.3.0 SDK

## Installation instructions

1. If you haven't got them already, install Visual Studio 2017, CMake, CUDA 10.x or 11.x and OptiX 7.3.0 from:
  * https://my.visualstudio.com/Downloads?q=visual%20studio%202017&wt.mc_id=o~msft~vscom~older-downloads
  * https://cmake.org/download/
  * https://developer.nvidia.com/cuda-toolkit
  * https://developer.nvidia.com/optix

2. Follow the instructions in the OptiX SDK to build the example files.
  * You should find installation instructions for different operating systems in the SDK directory where you installed OptiX 7.3.0.

3. Add the code from this repository to the examples in the SDK.
  * Checkout the code into a sub-directory of the SDK called 'optixGcodeViewer'.
  * Edit the CMakeLists.txt file in the main SDK directory to add the new sub-directory. Towards the end of the file you should see a list of add_subdirectory commands for each of the existing examples. Add the new code by adding this line:

    add_subdirectory( optixGcodeViewer       )

4. Re-run CMake to generate the files for optixGcodeViewer in your build directory.
  * If you're using the GUI run both 'Configure' and 'Generate' pointing to the existing Source and Build directories.
  * You should now find that optixGcodeViewer has been added into the build directory, and can be compiled together with the original examples.

