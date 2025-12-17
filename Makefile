# Compiler and flags
CXX     := g++
CXXFLAGS:= -Wall   
CXXFLAGS+= -Wextra
CXXFLAGS+= -std=c++17
CXXFLAGS+= -O2

# Target executable name
TARGET  := main

# Source files
SRCS    := main.cpp neural_network.cpp

# Object files (replace .cpp with .o)
OBJS    := $(SRCS:.cpp=.o)

# Default rule: do nothing
.DEFAULT_GOAL := help

help:
	@echo "Usage:"
	@echo "  make all   # build $(TARGET)"
	@echo "  make clean # remove objects and binary"

all: $(TARGET)

# Link step
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

# Compile steps
main.o: main.cpp main.h neural_network.h
	$(CXX) $(CXXFLAGS) -c main.cpp

neural.o: neural_network.cpp neural_network.h
	$(CXX) $(CXXFLAGS) -c neural_network.cpp

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)
