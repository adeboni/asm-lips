set OpenCV_DIR="C:\opencv\build"
rm -rf build
mkdir build
cd build
cmake -G "Visual Studio 12 2013" -D OpenCV_DIR=%OpenCV_DIR% ../src
msbuild ALL_BUILD.vcxproj
cd ..