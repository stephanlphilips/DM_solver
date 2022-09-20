CC =gcc
CXX=g++
RM=rm -f

ARMA_lib = /usr/local/Cellar/armadillo/11.2.3_1

CXXFLAGS = -g -fPIC -std=c++11 -fopenmp
CPPFLAGS = -I$(ARMA_lib)/include
LDFLAGS  = -L$(ARMA_lib)/lib 

LDLIBS   = -larmadillo -lgomp

SRCS= DM_solver_core.cpp memory_mgmnt.cpp math_functions.cpp python_c_interface.cpp
OBJS= $(subst .cpp,.o,$(SRCS))

c_libraries_DM_solver.so: $(OBJS)
	$(CXX) $(LDFLAGS) -dynamiclib -shared -o c_libraries_DM_solver.so $(OBJS) $(LDLIBS)
	
DM_solver_core.o: DM_solver_core.cpp DM_solver_core.h

memory_mgmnt.o: memory_mgmnt.cpp memory_mgmnt.h

math_functions.o: math_functions.cpp math_functions.h

python_c_interface.o: python_c_interface.cpp python_c_interface.h


clean:
	$(RM) $(OBJS)

clean-all:
	$(RM) $(OBJS) c_libraries_DM_solver.so