// This is a bit hacky, but we constrain the search space in this header file
// Statically-sized declarations of arrays in the code depend on this
// In most cases, this can be adapted to the program via dynamic allocation, user-supplied parameters, but I'm too lazy right now to bother

#define LOWER_BOUND 0
#define UPPER_BOUND 25

#define MAX_INSTRUCTIONS 50
#define MAX_INPUTS 25
#define REGISTERS 50

