CC = gcc
CFLAGS = -fPIC -Wall -Werror
LDFLAGS = -shared -lflash -lm
TARGET = build/libvoid.so
SRC = src/void.c
OBJ = $(SRC:.c=.o)
INCLUDE = src/void.h
INSTALL_DIR_LIB = /usr/lib
INSTALL_DIR_INCLUDE = /usr/include

all: $(TARGET)

$(TARGET): $(OBJ) $(INCLUDE) 
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c $(INCLUDE) 
	$(CC) $(CFLAGS) -c $< -o $@

install: $(TARGET)
	@if [ $(shell id -u) -ne 0 ]; then \
        echo "Error: You need to be root to install the library."; \
        exit 1; \
    fi
	sudo cp $(TARGET) $(INSTALL_DIR_LIB)
	sudo cp $(INCLUDE) $(INSTALL_DIR_INCLUDE)
	sudo ldconfig

uninstall:
	@if [ $(shell id -u) -ne 0 ]; then \
        echo "Error: You need to be root to uninstall the library."; \
        exit 1; \
    fi
	sudo rm $(INSTALL_DIR_LIB)/$(TARGET)
	sudo rm $(INSTALL_DIR_INCLUDE)/$(INCLUDE)
	sudo ldconfig

clean:
	rm -f $(TARGET) $(OBJ)

.PHONY: all install uninstall clean
