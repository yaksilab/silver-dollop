# Data structure for sub segmentation

The output data is stored two matrices, one for trace and one aligned masks.


## Trace data structure
	
```math
\begin{bmatrix}
C_1 & S_1 & x & y & t & d(t) & plane\# & class & f_1 & f_2 & ... & f_F \\ 
C_1 & S_2 & x & y & t & d(t) & plane\# & class & f_1 & f_2 & ... & f_F \\
... \\
... \\
C_2 & S_1 & x & y & t & d(t) & plane\# & class & f_1 & f_2 & ... & f_F \\
C_2 & S_2 & x & y & t & d(t) & plane\# & class & f_1 & f_2 & ... & f_F \\
... \\
... \\
C_N & S_1 & x & y & t & d(t) & plane\# & class & f_1 & f_2 & ... & f_F \\
C_N & S_2 & x & y & t & d(t) & plane\# & class & f_1 & f_2 & ... & f_F \\
... \\
... \\
\end{bmatrix}
```

Where:
- C: Cell ID
- S: Segment ID
- $(x,y)$: Center of mass  of the pixel coordinates
- $(t,d(t))$: `t` is the variable of the parameterized curve and `d(t)` is the distance between the subsegment of a cell and the `t`
value that the principal componente of the cell intersects the parameterized curve. So, `d(t)` is related to which subsegment number
- plane#: z coordinate
- class: class of the mask (1: complete cell `upper`, 2: complete cell `lower`, 3: process `upper` part, 4: process `lower`)	
- $f_1, f_2, ... f_F$: trace intensity across time points or frames


## Mask data structure

```math
\begin{bmatrix}
C_1 & S_1 & (x_1,y_1) & plane\# \\
C_1 & S_1 & (x_2,y_2) & plane\# \\
... \\
... \\
C_1 & S_2 & (x_1,y_1) & plane\# \\
C_1 & S_2 & (x_2,y_2) & plane\# \\
... \\
... \\
C_2 & S_1 & (x_1,y_1) & plane\# \\
C_2 & S_1 & (x_2,y_2) & plane\# \\
... \\
C_2 & S_2 & (x_1,y_1) & plane\# \\
C_2 & S_2 & (x_2,y_2) & plane\# \\
... \\
... \\
... \\
C_N & S_1 & (x_1,y_1) & plane\# \\
\end{bmatrix}
```

Where:

- $C$: Cell ID
- $S$: Segment ID
- $(x,y)$: pixel coordinates
- $plane\#$: z coordinate



