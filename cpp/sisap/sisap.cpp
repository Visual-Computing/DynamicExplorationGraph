#if defined(_WIN32)
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstdio>
#include <string>

// Forward declarations of the task entry functions
int run_task1(int argc, char* argv[]);
int run_task2(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <task1|task2> <hdf5_file_path> <mode_name> [options...]\n", argv[0]);
        return 1;
    }

    std::string task = argv[1];
    if (task == "task1") {
        // Shift arguments to left by 1: argv[1] becomes argv[0] of run_task1
        return run_task1(argc - 1, argv + 1);
    } else if (task == "task2") {
        // Shift arguments to left by 1: argv[1] becomes argv[0] of run_task2
        return run_task2(argc - 1, argv + 1);
    } else {
        std::fprintf(stderr, "Error: Unknown task '%s'. Choose 'task1' or 'task2'.\n", task.c_str());
        std::fprintf(stderr, "Usage: %s <task1|task2> <hdf5_file_path> <mode_name> [options...]\n", argv[0]);
        return 1;
    }
}
