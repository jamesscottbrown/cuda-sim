GSL_LIB = /cluster/soft/Linux_2.6_64/lib
GSL_INC = /cluster/soft/Linux_2.6_64/include

all : run_p53_ssa run_p53_sde

run_immdeath_ssa :
	cp immdeath_ssa.cu app.cu
	g++ -O3 -funroll-loops main_ssa_cpu.cpp -o main_ssa_cpu -I$(GSL_INC) -L$(GSL_LIB) -lgsl -lgslcblas
	#./main_ssa_cpu 100 16 41 1 res_ssa_immdeath.txt

run_immdeath_sde :
	cp immdeath_sde.cu app.cu
	g++ -O3 -funroll-loops main_sde_cpu.cpp -o main_sde_cpu -I$(GSL_INC) -L$(GSL_LIB) -lgsl -lgslcblas
	#./main_sde_cpu 100 16 41 1 res_sde_immdeath.txt

run_p53_ssa :
	cp p53_ssa.cu app.cu
	g++ -O3 -funroll-loops main_ssa_cpu.cpp -o main_ssa_cpu -I$(GSL_INC) -L$(GSL_LIB) -lgsl -lgslcblas
	#./main_ssa_cpu 100 16 41 1 res_ssa_p53.txt

run_p53_sde :
	cp p53_sde.cu app.cu
	g++ -O3 -funroll-loops main_sde_cpu.cpp -o main_sde_cpu -I$(GSL_INC) -L$(GSL_LIB) -lgsl -lgslcblas
	#./main_sde_cpu 100 16 41 1 res_sde_p53.txt

clean :
	rm -f main_ssa_cpu main_sde_cpu *.txt app.cu
