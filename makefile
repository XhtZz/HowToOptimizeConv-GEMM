## Compiler Flags
CPP	= $(COMPILER_PREF)g++ -mavx2 -mfma
CPPFLAGS = -O2 -fPIC -std=c++11

LOCAL_INC = -I./include
INCLUDE	  = $(LOCAL_INC)
############################################################
## Output Settings
TARGET_EXE_NAME	= compile
EXE_OUT		= $(TARGET_EXE_NAME)

SRCPATH		= ./
CXX_SRC		:= $(wildcard $(SRCPATH)*.cpp)
SRC_CXX 	:= $(notdir $(CXX_SRC))
CXX_OBJ		:= $(patsubst %.cpp,%.o,$(SRC_CXX))

OBJPATH		= ./
OBJS		:= $(patsubst %,$(OBJPATH)%, $(CXX_OBJ))

############################################################
## Make Commands
.PHONY: exe
exe: mkdirs exe_t

.PHONY: mkdirs
mkdirs:
	mkdir	-p $(OBJPATH)

.PHONY: exe_t
exe_t: $(OBJS) 
	$(CPP) $^ -o $(EXE_OUT)

$(OBJPATH)%.o: $(SRCPATH)%.cpp
	$(CPP) $(CPPFLAGS) $(INCLUDE) -c -o $@ $< 

.PHONY: test
test: 
	./$(TARGET_EXE_NAME) 

.PHONY: clean	
clean:
	rm $(OBJPATH)*.o $(TARGET_EXE_NAME) -f