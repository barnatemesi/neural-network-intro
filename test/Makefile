# =========================================================================
#   Unit Test Makefile
# =========================================================================

#We try to detect the OS we are running on, and adjust commands as needed
ifeq ($(OS),Windows_NT)
  ifeq ($(shell uname -s),) # not in a bash-like shell
	CLEANUP = del /F /Q
	MKDIR = mkdir
  else # in a bash-like shell, like msys
	CLEANUP = rm -f
	MKDIR = mkdir -p
  endif
	TARGET_EXTENSION=.exe
else
	CLEANUP = rm -f
	MKDIR = mkdir -p
	TARGET_EXTENSION=.out
endif

ENABLE_CLANG=0

ifeq ($(ENABLE_CLANG), 0)
	CXX_COMPILER=g++
else
	CXX_COMPILER=clang++
endif
ifeq ($(shell uname -s), Darwin) # this is macOS
CXX_COMPILER=clang++
endif

ROOT_DIR=..
TARGET_DIR=$(ROOT_DIR)/build

# Test defines (RELU needed for training tests where output values exceed tanh range)
CXX_TEST_DEFS=-DDEBUG
CXX_TEST_DEFS+=-DACTIVATION_FN_IS_RELU

CXXFLAGS=-Wall
CXXFLAGS+=-Wextra
CXXFLAGS+=-std=c++17
CXXFLAGS+=-Og
CXXFLAGS+=-Wno-maybe-uninitialized

TARGET_BASE_TEST=test_runner
TARGET_TEST = $(TARGET_BASE_TEST)$(TARGET_EXTENSION)

CXX_UNIT_TEST_SOURCES = \
$(ROOT_DIR)/src/neural_network.cpp \
$(ROOT_DIR)/src/misc.cpp \
test_neural_network.cpp \
test_misc.cpp \

INC_DIRS = \
-I$(ROOT_DIR)/inc \
-I$(ROOT_DIR)/eigen \

CATCH2_LIBS = -lCatch2Main -lCatch2

test:
	$(MKDIR) $(TARGET_DIR)
	$(CXX_COMPILER) $(CXX_TEST_DEFS) $(CXXFLAGS) $(INC_DIRS) $(CXX_UNIT_TEST_SOURCES) $(CATCH2_LIBS) -o $(TARGET_DIR)/$(TARGET_TEST)
	./$(TARGET_DIR)/$(TARGET_TEST) --reporter compact

clean:
	$(CLEANUP) $(TARGET_DIR)/$(TARGET_TEST)

.PHONY: test clean
