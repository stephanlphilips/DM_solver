CC =gcc
CXX=g++
RM=rm -f

CXXFLAGS = -g -fPIC -std=c++11 -fopenmp
CPPFLAGS = -I/usr/local/Cellar/armadillo/11.0.0/include
LDFLAGS  = -larmadillo -lgomp
LDLIBS   = -L/usr/local/Cellar/armadillo/11.0.0/lib 

SRCS= DM_solver_core.cpp memory_mgmnt.cpp math_functions.cpp python_c_interface.cpp
OBJS= $(subst .cpp,.o,$(SRCS))

c_libraries_DM_solver.so: $(OBJS)
	echo $(OBJS)
	$(CXX) $(LDFLAGS) -dynamiclib -shared -o c_libraries_DM_solver.so $(OBJS) $(LDLIBS)

DM_solver_core.o: DM_solver_core.cpp DM_solver_core.h

memory_mgmnt.o: memory_mgmnt.cpp memory_mgmnt.h

math_functions.o: math_functions.cpp math_functions.h

python_c_interface.o: python_c_interface.cpp python_c_interface.h


clean:
	$(RM) $(OBJS) c_libraries_DM_solver.so