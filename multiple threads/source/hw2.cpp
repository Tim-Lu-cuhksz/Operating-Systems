#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 
#define LENGTH 15

pthread_mutex_t mutex;

typedef enum {
	WIN, LOSE, EXIT, NONE // NONE: default status
} status_t;

status_t STATUS = NONE;

struct Node{
	int x , y;
} frog ; 

char map[ROW+10][COLUMN] ; 
int bar_len[ROW];

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

void print_status(status_t s) {
	printf("\033[H\033[2J");
	if (s == LOSE) {			
		printf("You lose the game!!\n");
	} else if (s == EXIT) {
		printf("You exit the game.\n");
	} else if (s == WIN){
		printf("You win the game!!\n");
	} else printf("Invalid status.");
}

void *logs_move( void *t ){
	int* ptr = (int *)t;
	int row = *ptr; // Row index
	int hasFrog = 0; // Return to 0 when frog leaves
	
	// get the position of the log
	srand(row+(int)time(0));
	int col = rand() % (COLUMN-1); // Start column index
	// srand(time(0));
	// int LENGTH = rand() % (15);
	// critical region starts
	pthread_mutex_lock(&mutex);
	// Draw the log on the map
	for(int j = col; j < (col + LENGTH + COLUMN-1)%(COLUMN-1); ++j ) {			
		map[row][j] = '=' ;	
	}	
	pthread_mutex_unlock(&mutex);
	
	// We should apply condition here to break the loop
	// rather than using "break" in the if statement
	while (STATUS == NONE)
	{
		pthread_mutex_lock(&mutex);
		if (frog.x == row) {
		// The frog is not on the log
			if ((frog.y < col && frog.y >= LENGTH-1) ||
				(frog.y >= col+LENGTH && (COLUMN-2-frog.y) >= LENGTH-1))
			{
				STATUS = LOSE;
			}
			hasFrog = 1;
		} else hasFrog = 0;

		/*  Move the logs  */
		// critical region begins				
		if (!hasFrog) {
			// Left
			if (row % 2 != 0) {				
				map[row][(col + LENGTH + COLUMN-2)%(COLUMN-1)] = ' ';
				map[row][(col + COLUMN-2)%(COLUMN-1)] = '=' ;
				col = (col+ COLUMN-2)%(COLUMN-1);
				// map[row][(col + COLUMN-2)%(COLUMN-1)] = 
				// 		map[row][(col + LENGTH + COLUMN-2)%(COLUMN-1)];
				// map[row][(col + LENGTH + COLUMN-2)%(COLUMN-1)] = ' ';
				// col = (col+ COLUMN-2)%(COLUMN-1);

			} else {
				// Right
				map[row][(col + LENGTH + COLUMN-1)%(COLUMN-1)] = '=';
				map[row][(col)] = ' ' ;
				col = (col+1)%(COLUMN-1);
			}
		// There is a frog on the log	
		} else {
			// Left
			if (row % 2 != 0) {
				map[row][frog.y] = map[row][frog.y+1];
				map[row][frog.y-1] = '0';

				map[row][(col + LENGTH + COLUMN-2)%(COLUMN-1)] = ' ';
				if (col != frog.y) {
					map[row][(col + COLUMN-2)%(COLUMN-1)] = '=' ;
				}				
				col = (col + COLUMN-2)%(COLUMN-1);
				frog.y -= 1;
			} else { // Right
				map[row][frog.y] = map[row][frog.y-1];
				map[row][frog.y+1] = '0';
				map[row][(col)] = ' ' ;
				if ((col+LENGTH-1) != frog.y) {
					map[row][(col + LENGTH + COLUMN-1)%(COLUMN-1)] = '=';
				}								
				col = (col+1)%(COLUMN-1);
				frog.y += 1;
			}
		}	
		/*  Check keyboard hits, to change frog's position or quit the game. */
		if (kbhit()) {
			char dir = getchar();
			/* move left  */
			if (dir == 'a' || dir == 'A') {
                if (frog.y > 0) {
                    map[frog.x][frog.y] = map[frog.x][frog.y-1];
				    map[frog.x][frog.y-1] = '0';
				    frog.y -= 1;
                }				
			}
			/* move right */
			if (dir == 'd' || dir == 'D') {
                if (frog.y < COLUMN - 2) {
                   map[frog.x][frog.y] = map[frog.x][frog.y+1];
				    map[frog.x][frog.y+1] = '0';
				    frog.y += 1; 
                }				
			}
			/* move up */
			if (dir == 'w' || dir == 'W') {
				if (frog.x == ROW) {
					map[frog.x][frog.y] = '|';
				} else map[frog.x][frog.y] = '=';
				// When moving up, x_coordinator decreases
				map[frog.x-1][frog.y] = '0';
				frog.x -= 1;
			}
			/* move down */
			if (dir == 's' || dir == 'S') {
				if (frog.x < ROW) {
					map[frog.x][frog.y] = '=';
					map[frog.x+1][frog.y] = '0';
					frog.x += 1;
				}				
			}
			if (dir == 'q' || dir == 'Q') {
				STATUS = EXIT;
			}
		}
		/*  Print the map on the screen  */
		printf("\033[H\033[2J");
		for( int i = 0; i <= ROW; ++i){			
			puts( map[i] );			
		}
		/*  Check game's status  */
		if (frog.y < 0 || frog.y >= COLUMN-1) {
			STATUS = LOSE;
		}
		if (frog.x == 0) {
			STATUS = WIN;
		}
		//if (STATUS != NONE) pthread_testcancel();
		pthread_mutex_unlock(&mutex);
		usleep(50000);
	}
	pthread_exit(NULL);
}

int main( int argc, char *argv[] ){

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j ) {	
		map[ROW][j] = '|' ;
		map[0][j] = '|' ;
	}

	//frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	frog.x = ROW; frog.y = (COLUMN-1)/2;
	map[frog.x][frog.y] = '0' ; 

	//Print the map into screen
	printf("\033[H\033[2J");
	for( i = 0; i <= ROW; ++i)	
		puts( map[i] );
		
	/*  Create pthreads for wood move and frog control.  */
	if (pthread_mutex_init(&mutex,NULL) != 0) {
		printf("Mutex init has failed.\n");
		return 1;
	}

	int row_index[ROW-1];
	for (int k = 0; k < ROW-1; k++) row_index[k] = k+1;
    pthread_t ptid[ROW-1];
	for (int k = 0; k < ROW-1; k++) {
		pthread_create(&ptid[k], NULL, logs_move, &row_index[k]);
	}
	for (int k = 0; k < ROW - 1; k++) {
		pthread_join(ptid[k],NULL);
	}

	pthread_mutex_destroy(&mutex);
	/*  Display the output for user: win, lose or quit.  */
	print_status(STATUS);
	pthread_exit(NULL);

	return 0;
}