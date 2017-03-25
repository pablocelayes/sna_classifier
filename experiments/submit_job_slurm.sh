#!/bin/bash
#
# Author:  Carlos E. Budde
# Date:    03.05.2015
# License: GPLv3
#
### Las líneas #SBATCH configuran los recursos de la tarea
### (aunque parezcan estar comentadas)

### Cola de trabajos a usar
### Opciones en mendieta: capacity, capability, gpu)
### Opciones en jupiterace: batch
### Opciones en zx81: batch
#SBATCH --partition=batch

### Nombre de la tarea
#SBATCH --job-name=<FILLME!>

### Cantidad de nodos a usar
#SBATCH --nodes=1

### Procesos por nodo
#SBATCH --ntasks-per-node=1

### Cores visibles por nodo
### En mendieta: <= 16
### En jupiterace: <= 6
### En zx81: <= 12
#SBATCH --cpus-per-task=12

### Tiempo de ejecución. Formato dias-horas:minutos. Máximo: tres días.
#SBATCH --time 2-10:00:00

### Lanzado de la tarea
srun /bin/bash main.sh

exit 0

