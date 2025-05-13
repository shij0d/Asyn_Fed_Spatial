# Thread Control Logic with Explanations

## Thread Control Logic

\[
\begin{aligned}
\textbf{(1)~OMP}_{\text{effective}} &=
\begin{cases}
0, & \text{if } \text{OMP}_{\text{set}} > \text{NUMEXPR}_{\text{max}} \\
\text{NUMEXPR}_{\text{default}}, & \text{if } \text{OMP}_{\text{set}} \text{ is unset} \\
\text{OMP}_{\text{set}}, & \text{otherwise}
\end{cases} \\
&\quad\text{• Effective number of OpenMP threads used.} \\
&\quad\text{• Disabled (0) if too many threads are requested (exceeds the number of threads NumExpr detects).} \\
&\quad\text{• Defaults to the number of threads NumExpr detects if OMP threads are unset.} \\
\\
\textbf{(2)~MKL}_{\text{effective}} &=
\begin{cases}
\text{OMP}_{\text{set}}, & \text{if } \text{MKL}_{\text{set}} \text{ is unset} \\
\text{MKL}_{\text{set}}, & \text{otherwise}
\end{cases} \\
&\quad\text{• Actual number of MKL threads used.} \\
&\quad\text{• Defaults to OMP setting if MKL setting is missing.} \\
\\
\textbf{(3)~Threads}_{\text{total}} &= \text{OMP}_{\text{effective}} + \text{MKL}_{\text{effective}} + \text{Threads}_{\text{other}} \\
&\quad\text{• Total thread usage by the program.} \\
&\quad\text{• Includes other threaded libraries (e.g., NumExpr, OpenBLAS, custom pools).} \\
\\
\textbf{(4)~MKL Usage (CPU-bound tasks)} & \quad \text{MKL uses more CPU for CPU-bound tasks. It is optimized for high-performance computations.} \\
&\quad\text{• Best for heavy numerical or linear algebra tasks.} \\
\\
\textbf{(5)~Excessive Threads (CPU contention)} & \quad \text{Too many threads can lead to CPU contention and inefficiencies.} \\
&\quad\text{• If total thread count exceeds available cores, threads compete for CPU resources, reducing performance.} \\
\\
\textbf{(6)~Hyperthreading Effect} & \quad \text{Hyperthreading (logical cores) can make the procedure faster under certain conditions.} \\
&\quad\text{• Helps in better utilization of CPU resources when there are idle cycles.} \\
&\quad\text{• Optimal performance depends on the nature of the task. CPU-bound tasks benefit less from hyperthreading.}
\end{aligned}
\]

## Explanation of Updated Insights

1. **MKL Usage (CPU-bound tasks)**:  
   Intel's **MKL** (Math Kernel Library) utilizes more CPU resources in our case.

2. **Excessive Threads (CPU contention)**:  
   If the total number of threads exceeds the available CPU cores, **CPU contention** occurs, where threads compete for processing time. 

3. **Hyperthreading Effect**:  
   **Hyperthreading** (logical cores) can improve performance by better utilizing CPU.
