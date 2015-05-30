# asm-lips
GPU-Accelerated Lip-Tracking Library
============
Info:
---------------------
This is a general purpose lip-tracking library. 
It uses the Active Shape Model algorithm, specifically the one detailed in [Mastering OpenCV](http://www.amazon.com/Mastering-OpenCV-Practical-Computer-Projects/dp/1849517827). 
This code is largely the same code as from Mastering OpenCV, but with some optimizations using CUDA.
This library can be run on OS X, Linux, Windows, and Android (using JNI).

Prerequisites:
---------------------
CMake and OpenCV 2.4.9 or 2.4.10 is installed. You also need CUDA 6.5 or newer installed if you want to compile it with CUDA support.


Installation:
---------------------
### Building the project using CMake from the command-line:

### Linux:
>    export OpenCV_DIR="~/OpenCV/build"
>    mkdir build
>    cd build
>    cmake -D OpenCV_DIR=$OpenCV_DIR ../src
>    make 

### OS X (Xcode):
>    export OpenCV_DIR="~/OpenCV/build"
>    mkdir build
>    cd build
>    cmake -G Xcode -D OpenCV_DIR=$OpenCV_DIR ../src
>    open OPENCV_HOTSHOTS.xcodeproj

### Windows (MS Visual Studio):
>    set OpenCV_DIR="C:\OpenCV\build"
>    mkdir build
>    cd build
>    cmake -G "Visual Studio 12 2013" -D OpenCV_DIR=%OpenCV_DIR% ../src
>    start OPENCV_HOTSHOTS.sln 
    
- A static library will be written to the "lib" directory.
- The execuables can be found in the "bin" directory.

Usage:
---------------------
