CC=g++

SOURCEDIR=src/
BUILDDIR=build/

DATADIR=\"data/\"

HEADERS=$(wildcard $(SOURCEDIR)*.h)
SOURCES=$(wildcard $(SOURCEDIR)*.cpp)

OBJECTS= $(addprefix $(BUILDDIR),$(notdir $(SOURCES:.cpp=.o)))

EXECUTABLE=$(BUILDDIR)mosse_filter

PFLAGS=-DDATA_DIRECTORY=$(DATADIR)
CFLAGS=-ggdb -Wall -I./ -I/usr/local/includes
LDFLAGS=-L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_flann -lopencv_legacy -lopencv_ml -lopencv_features2d -lfftw3


all: $(SOURCES) $(EXECUTABLE)
	@echo Complited!

$(BUILDDIR)%.o: $(SOURCEDIR)%.cpp
	@echo Compiling $< into $@ ...
	@$(CC) $(PFLAGS) $(CFLAGS) -c $< -o $@

$(EXECUTABLE): $(OBJECTS) 
	@echo Making executable file $@...
	@$(CC) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	@echo Deleting files $(OBJECTS) $(EXECUTABLE)...
	@rm $(OBJECTS) $(EXECUTABLE)
