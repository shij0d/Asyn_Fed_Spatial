### 1. Simulated Algorithm

Previously, our simulated algorithm consisted of two main classes:  
- **_Server_**: Contains functions that run on the server.  
- **_Worker_**: Contains functions that run on the worker.  

For communication simulation, the **_Server_** directly manipulates the **_Worker_** and retrieves the results.

---

### 2. MPI-Based Algorithm

In contrast, for the MPI-based algorithm, the **_Server_** cannot directly manipulate the **_Worker_**.  

Explicitly defining the actions of the **_Worker_** would lead to complex code.  

To simplify this, we adopt a messaging mechanism where:  
1. The **_Server_** sends a message containing commands specifying what the **_Worker_** should do.  
2. The **_Worker_** continuously waits for commands from the **_Server_** using a `while` loop.  

This approach results in a much simpler and more maintainable implementation.


We use multithread in both **_Server_** and **_Worker_** to let computation and communication can happen simultaneously:
- Communication Process: for the communication
- Computation Process: for the computation


Without multithreading: local computation blocking receiving and sending
With multithreading: the context can be changed to receiving or sending 

