#include <mpi.h>
#include <iostream>
#include <random>
#include <cmath>
#include <fstream>

void mpi_algorithm(int Mmax, unsigned int seed) {
    int rank, size;

    // Initialize an MPI session
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the number of processes

    int N = size;  // Total number of processes

    // Output the number of processes and the rank of the current process
    std::cout << "N = " << N << std::endl;
    std::cout << "Core = " << rank << std::endl;

    // Initialize random number generator
    std::default_random_engine generator(seed + rank);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    if (rank == 0) {
        // Variables for storing the sum of squares and sum of fourth powers
        double E = 0.0, V = 0.0;
        int M = 0;  // Counter for the number of samples
        double message[3] = {E, V, static_cast<double>(M)};

        // Pick a random target process to send the initial message
        int target = 1 + (rand() % (N - 1));
        MPI_Send(message, 3, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);

        while (true) {
            // Receive message from any source
            MPI_Recv(message, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            E = message[0];
            V = message[1];
            M = static_cast<int>(message[2]);

            // Check if the number of samples has reached Mmax
            if (M == Mmax) {
                // Output results to a file and to the standard error
                std::ofstream outfile("output.txt");
                std::cerr << "E/Mmax= " << E / Mmax << std::endl;
                std::cerr << "V/Mmax - E^2/Mmax^2= " << V / Mmax - E * E / (Mmax * Mmax) << std::endl;
                outfile << "E/Mmax= " << E / Mmax << std::endl;
                outfile << "V/Mmax - E^2/Mmax^2= " << V / Mmax - E * E / (Mmax * Mmax) << std::endl;
                outfile.close();
                break;
            } else {
                // Generate new samples using the Box-Muller transformation
                double u1 = distribution(generator);
                double u2 = distribution(generator);
                double x1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                double x2 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

                // Update E and V
                E += x1 * x1 + x2 * x2;
                V += (x1 * x1 + x2 * x2) * (x1 * x1 + x2 * x2);
                M++;

                // Pick a new target process randomly
                int target;
                do {
                    target = rand() % N;
                } while (target == rank);

                // Send updated message to the new target
                message[0] = E;
                message[1] = V;
                message[2] = static_cast<double>(M);
                MPI_Send(message, 3, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
            }
        }

        // Send a finalization message to all other processes
        for (int i = 1; i < N; ++i) {
            MPI_Send(message, 3, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        }
    } else {
        // Processes other than rank 0
        double message[3];
        while (true) {
            MPI_Status status;
            // Receive message from any source
            MPI_Recv(message, 3, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            // Break loop if the finalization message is received
            if (status.MPI_TAG == 1) {
                break;
            }

            double E = message[0];
            double V = message[1];
            int M = static_cast<int>(message[2]);

            // Check if the number of samples is less than Mmax
            if (M < Mmax) {
                // Generate new samples using the Box-Muller transformation
                double u1 = distribution(generator);
                double u2 = distribution(generator);
                double x1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                double x2 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

                // Update E and V
                E += x1 * x1 + x2 * x2;
                V += (x1 * x1 + x2 * x2) * (x1 * x1 + x2 * x2);
                M++;

                // Pick a new target process randomly
                int target;
                do {
                    target = rand() % N;
                } while (target == rank);

                // Send updated message to the new target
                message[0] = E;
                message[1] = V;
                message[2] = static_cast<double>(M);
                MPI_Send(message, 3, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
            } else {
                // If M is equal to Mmax, send message back to rank 0
                MPI_Send(message, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    // Finalize the MPI session
    MPI_Finalize();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage = " << argv[0] << " <N> <Mmax>" << std::endl;
        return 1;
    }

    int Mmax = std::stoi(argv[1]);
    std::cout << "Mmax = " << Mmax << std::endl;
    unsigned int seed = 123;

    // Call the mpi_algorithm function
    mpi_algorithm(Mmax, seed);

    return 0;
}
