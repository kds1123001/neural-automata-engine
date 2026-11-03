CC       ?= cc
OPT      ?= -O3
ARCH     ?= -march=native
WARN     := -Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes

CPPFLAGS := -Iinclude
CFLAGS   := -std=c99 $(OPT) $(ARCH) $(WARN)

ifeq ($(SAN),1)
CFLAGS   += -g -fsanitize=address,undefined -fno-omit-frame-pointer
LDFLAGS  += -fsanitize=address,undefined
endif

ifeq ($(OMP),1)
CFLAGS   += -fopenmp
LDFLAGS  += -fopenmp
endif

SRC      := $(wildcard src/*.c) $(wildcard src/rules/*.c)
OBJ      := $(patsubst src/%.c,build/obj/%.o,$(SRC))
DEPS     := $(OBJ:.o=.d)
HEADERS  := $(wildcard include/nae/*.h)
TEST_SRC := $(wildcard tests/test_*.c)
TEST_BIN := $(patsubst tests/%.c,build/tests/%,$(TEST_SRC))

.PHONY: all lib test pytest check bench clean

all: lib

lib: build/libnae.a build/libnae.so

build/libnae.a: $(OBJ)
	ar rcs $@ $^

build/libnae.so: $(SRC) bindings/nae_capi.c $(HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared $(SRC) bindings/nae_capi.c -o $@ $(LDFLAGS)

build/obj/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

build/tests/%: tests/%.c tests/check.h build/libnae.a
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< build/libnae.a -o $@ $(LDFLAGS)

test: $(TEST_BIN)
	@for t in $(TEST_BIN); do echo "run $$t"; ./$$t || exit 1; done

pytest: build/libnae.so
	python3 -m unittest discover -s tests -p "test_*.py" -v

check: test pytest

bench: build/libnae.so
	python3 scripts/bench.py

clean:
	rm -rf build

-include $(DEPS)
