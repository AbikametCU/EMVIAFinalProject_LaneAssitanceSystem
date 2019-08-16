# EMVIAFinalProject_LaneAssitanceSystem
The code detects cars, lanes, pedestrain which can be further used for autonmous vehicles as well as uses the purpose of multithreading for achievement of highr FPS as well as using Haar cascades for lane detection and HOG features for pedestrain detection.

compilation of the .cpp: g++ `pkg-config --cflags opencv` testfile.cpp `pkg-config --libs opencv` -lpthread -lrt -o testfile
