# Frog Crossing a River

## Design
### 1.1	Multi-threads Creation
Since there are in total nine logs moving continually from right to left or vice versa, nine threads have been implemented to move each log. To realize that, we first need to create an array of pthread IDs to store them while initialization. A for loop, along with  the invocation of pthread_create(), will create nine threads to run the targeted function.  Note that the attribute of the pthread is set to default value NULL and arguments indicating the row index of each log are passed to it as well. The row index will be later used to determine the moving direction of the logs (log with odd row number moves leftwards, while that with even number moves rightwards). The function to be executed by the threads is logs_move() which is introduced in section 1.3.
### 1.2	Logs Generation
For a log with some specific row index, for example 3, the key information to generate the log is the starting column as well as the length of the log. Here, we set the default value of length of each log to be 15. As for the starting column, srand() and rand() are applied to generate the random position of the column number. To attain randomicity, time(0) is passed to srand() so that we can obtain the random number. The key to avoiding potential bugs here is to mod the random number with the length of the game board. It is always of utmost importance to beware of the borders and constraints of the game and we should try to keep things in order. Once obtaining the starting column of the log, we can draw the log by assigning “=” to the positions in the map. Also, we ought to mod the column index when assigning to stay away from any malfunctioning of the game as further as possible.
### 1.3	Logs Movement
Before moving the log in a row, we first check if the frog is on the same row as the log or not because the condition of the frog may affect the behavior of the log. If a frog appears in the same row and the position of the frog is not on the frog, then the game status will be LOSE and the game is immediately terminated. If a frog is on the log, it will move with the log on the river. The algorithm to move the log with or without the frog is not hard to understand which is similar to a snake game. That is, suppose the row index and starting column of the log is three and five respectively (map[3][5]), then it should move leftwards as the row index is an odd number. To move the object in the same row, we assign map[3][5]  to map[3][4] which contains nothing at the moment. By the same means, we apply the method to iteratively until reaching the end of the log. Again, when moving, careful considerations are taken to deal with boundary conditions and we use mod to tackle the problem. Note that every time we first clear the screen by printf("\033[H\033[2J") and print out the map so that it feels like the log is moving.
### 1.4	Frog Movement
The implementation of frog moving between logs can be carried within the function logs_move(). In other words, each thread is able to carry out the movement by itself. Mutex is used to avoid the interplay between two different threads. More details of mutex are covered in section 1.5. The frog does not take any action until the program receives keyboard hits from the player. The implementation of detection of the hits is provided and we only focus on how to move the frog after pressing the keyboard. One simple way to move the frog is to change the coordinates of it, and the new status will be refreshed in milliseconds.
### 1.5	Critical Region
To reduce the risk of interplay between different threads, we need to protect the data (global variables) from being accessed by two or three threads. The method of “mutually exclusive” can assist us to cope with the situation by calling pthread_mutex_init().  Before making use of global variables or shared data, we apply pthread_mutex_lock() to lock it. After that, pthread_mutex_unlock()  is invoked to unlock the situation.

## Environment
### Linux Version: Ubuntu 16.04
### Linux Kernel Version: 4.15.0

## Complile and Execute
```
# compile
make
# clear 
make clear
# run the program
./hw2
```

## How To Play
Use "w" "s" "a" "d" on the keyboard to move the frog. Hit "q" to quit the game.