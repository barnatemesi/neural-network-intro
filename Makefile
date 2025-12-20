# =========================================================================
#   Unity - A Test Framework for C
#   ThrowTheSwitch.org
#   Copyright (c) 2007-24 Mike Karlesky, Mark VanderVoord, & Greg Williams
#   SPDX-License-Identifier: MIT
# =========================================================================

# define firmware version
VERSION_MAJOR = 0
VERSION_MINOR = 0
VERSION_PATCH = 0

ENABLE_CLANG=0

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

TARGET_DIR=build

ifeq ($(ENABLE_CLANG), 0)
	CXX_COMPILER=g++
else
	CXX_COMPILER=clang++
endif
ifeq ($(shell uname -s), Darwin) # this is macOS
CXX_COMPILER=clang++
endif

# C defines
CXX_DEFS=-DDEBUG\
CXX_DEFS+=-DDEBUG_TRAIN \

CXXFLAGS=-Wall
CXXFLAGS+=-Wextra
CXXFLAGS+=-std=c++17
CXXFLAGS+=-Og
CXXFLAGS+=-Wno-maybe-uninitialized

TARGET_BASE1=program

TARGET1 = $(TARGET_BASE1)$(TARGET_EXTENSION)

C_TEST_SOURCES = \
src/main.cpp \
src/neural_network.cpp \

INC_DIRS = \
-Iinc \
-Ieigen \

LIBS = -lc -lm

SYMBOLS = -DUNITY_FIXTURE_NO_EXTRAS

help:
	@echo "Usage:" \
    "  make all   # build $(TARGET)" \
    "  make clean # remove objects and binary" \

all: clean default

# build and test
default:
	$(MKDIR) $(TARGET_DIR)
	$(CXX_COMPILER) $(C_DEFS) $(CXXFLAGS) $(INC_DIRS) $(SYMBOLS) $(C_TEST_SOURCES) $(LIBS) -o $(TARGET_DIR)/$(TARGET1)

clean:
	$(CLEANUP) $(TARGET_DIR)/$(TARGET1)

ci: CFLAGS += -Werror
ci: default