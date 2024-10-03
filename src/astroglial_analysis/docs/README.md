# Data structure for sub segmentation

The output data is stored in a dictionary with keys `trace` and `coords`.

$$Data = \{trace: [...], coords: [...]\}$$

`trace` is a 3D matrix with shape.
$$trace.shape = (C, S, F)$$

Where
- $C$ is the number of cells
- $S = \{S_1,S_2,...,S_n\}$ contains the number subcells for each cell
- $F$ is the number of time frames

`coords` is a 3D array with shape.
$$coords.shape = (C, S, 2)$$


`trace` is a 3D matrix 
```math
\begin{bmatrix} C 1:\begin{bmatrix} i_{111} & i_{112} &...&i_{11F} \\\ 
i_{121} & i_{122} &...&i_{12F} \\\
... \\\
... \\\
\end{bmatrix} \\\ \newline
C2:\begin{bmatrix} i_{211} & i_{212} &...&i_{21F} \\\
i_{221} & i_{222} &...&i_{22F} \\\
... \\\
... \\\
\end{bmatrix} \\\ \newline
... \\\ \newline
\end{bmatrix}
```


`Coords` is a list of lists of lists of tuples
```math
\begin{bmatrix} 
\begin{bmatrix} [(x, y) & (x, y) & ... \space],\\\
[(x, y) & (x, y) & ... \space],\\\
... \\\
... \\\
\end{bmatrix}, \\\ \newline
\begin{bmatrix} [(x, y) & (x, y) & ... \space],\\\
[(x, y) & (x, y) & ... \space],\\\
... \\\
... \\\
\end{bmatrix}, \\\ \newline
... \\\ \newline
\end{bmatrix}
```

The first index is the cell index, the second index is the subcell index and the third index is the coordinate index.

Each cell has different number of subcells and each subcell has different number of coordinates.