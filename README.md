
## Compilation
To compile the solution file, use the following code:
```
mpicc src.c -o src.exe -lm
```
 
## Execution
The executable will be named "src.exe". Now, to run the executable file using MPI, use the code shown below:
```
mpirun -np 8 ./src.exe 10 2 3 7 989
```
The last five numbers are input parameters defined by the assignment.

