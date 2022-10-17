TARGETS = hw2

CROSS_TOOL = 
CC_CPP = $(CROSS_TOOL)g++
CC_C = $(CROSS_TOOL)gcc

CFLAGS = -Wall -g -Werror -pthread

all: clean $(TARGETS)

$(TARGETS):
	$(CC_CPP) $(CFLAGS) $@.cpp -o $@

clean:
	rm -f $(TARGETS)