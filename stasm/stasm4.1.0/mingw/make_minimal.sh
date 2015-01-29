# make_minimal.sh

# -Wno-long-long        disables the "long long warnings" for the OpenCV headers
# -Wno-unused-parameter allows virtual func defs that don't use all params
# -Wno-unknown-pragmas  allows OpenMP pragmas without complaint

g++ -o minimal\
  -O3 -DMOD_1 -Wall -Wextra -pedantic\
 -Wno-long-long -Wno-unused-parameter -Wno-unknown-pragmas\
 -Wstrict-aliasing\
 -I/usr/local/include -I../stasm  -I../apps\
 -L/usr/local/lib/ \
 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect\
  ../apps/minimal.cpp ../stasm/*.cpp ../stasm/MOD_1/*.cpp
