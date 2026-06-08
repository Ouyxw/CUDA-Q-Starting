import cudaq

@cudaq.kernel
def kernel(qubit_count: int):
    # Allocate our qubits.
    qvector = cudaq.qvector(qubit_count)
    # Place the first qubit in the superposition state.
    h(qvector[0])
    # Loop through the allocated qubits and apply controlled-X, 
    # or CNOT, operations between them.
    for qubit in range(qubit_count - 1):
        x.ctrl(qvector[qubit], qvector[qubit + 1])
    # Measure all qubits in the Z-basis.
    mz(qvector)

### Bell state preparation and measurement.
qubit_count = 2
print(cudaq.draw(kernel, qubit_count))
results = cudaq.sample(kernel, qubit_count, shots_count=10000)
# Should see a roughly 50/50 distribution between 00 and 11 states.
print("Measurement results of Bell state:" + str(results))
most_probable_result = results.most_probable()
probability = results.probability(most_probable_result)
print("Most probable result: " + most_probable_result)
print("Measured with probability: " + str(probability), end="\n\n")

### GHZ state preparation and measurement.
qubit_count = 3
print(cudaq.draw(kernel, qubit_count))
results = cudaq.sample(kernel, qubit_count, shots_count=10000)
# Should see a roughly 50/50 distribution between 000 and 111 states.
print("Measurement results of GHZ state:" + str(results))
most_probable_result = results.most_probable()
probability = results.probability(most_probable_result)
print("Most probable result: " + most_probable_result)
print("Measured with probability: " + str(probability), end="\n\n")