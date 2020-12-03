#ifndef WALKER_DEFS_H
#define WALKER_DEFS_H

#ifndef DIRECTION_ENUM
#define DIRECTION_ENUM
typedef enum Direction
{
    None = 0,
    North = 1,
    West = 2,
    South = 3,
    East = 4,
    Up = 5,
    Down = 6
} direction;
#endif

#define NORTH 1
#define WEST 2
#define SOUTH 3
#define EAST 4
#define UP 5
#define DOWN 6

#define LOG_XIRATE false
#define WALKER_INITIAL_ENERGY 1.0
#define WALKER_DEFAULT_RHO 20.0

#endif