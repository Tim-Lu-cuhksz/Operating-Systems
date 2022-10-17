#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]){
	int status;
	/* fork a child process */
	printf("Process starts to fork\n");
	pid_t pid = fork();
	/* execute test program */ 
	if (pid < 0) {
		perror("Fork failed.");
	}
	if (pid == 0) {

		int i;
		char *arg[argc];

		printf("I'm the Child Process, my pid = %d\n",getpid());
		for (i=0; i<argc-1; i++){
			arg[i] = argv[i+1];
		}
		arg[argc-1] = NULL;

		printf("Child process starts to execute test program:\n");
		execve(arg[0], arg, NULL);
		//exit(0);
	}	
	printf("I'm the Parent Process, my pid = %d\n",getpid());

	/* wait for child process terminates */
	waitpid(pid, &status, WUNTRACED);
	printf("Parent process receives the SIGCHLD signal\n");

	/* check child process'  termination status */
	if (WIFEXITED(status)) {
		printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
	} 
	else if (WIFSIGNALED(status)) {
		if (WTERMSIG(status) == 1) {
			printf("Child process gets SIGHUP signal\n");
			printf("Child process is hung up\n");
		}
		else if (WTERMSIG(status) == 2) {
			printf("Child process gets SIGINT signal\n");
			printf("Child process is interrupted\n");
		}
		else if (WTERMSIG(status) == 3) {
			printf("Child process gets SIGQUIT signal\n");
			printf("Child process is quitted\n");
		}
		else if (WTERMSIG(status) == 4) {
			printf("Child process gets SIGILL signal\n");
			printf("Child process attempts to execute an illegal instruction\n");
		}
		else if (WTERMSIG(status) == 5) {
			printf("Child process gets SIGTRAP signal\n");
			printf("Child process is trapped\n");
		}
		else if (WTERMSIG(status) == 6) {
			printf("Child process gets SIGABRT signal\n");
			printf("Child process is aborted\n");
		}
		else if (WTERMSIG(status) == 7) {
			printf("Child process gets SIGBUS signal\n");
			printf("Child process reports a bus error\n");
		}
		else if (WTERMSIG(status) == 8) {
			printf("Child process gets SIGFPE signal\n");
			printf("Child process reports an error in arithmetic operation\n");
		}
		else if (WTERMSIG(status) == 9) {
			printf("Child process gets SIGKILL signal\n");
			printf("Child process is killed\n");
		}
		else if (WTERMSIG(status) == 11) {
			printf("Child process gets SIGSEGV signal\n");
			printf("Child process makes a segmentation fault\n");
		}
		else if (WTERMSIG(status) == 13) {
			printf("Child process gets SIGPIPE signal\n");
			printf("Child process attempts to write a pipe without a process connected to the other end\n");
		}
		else if (WTERMSIG(status) == 14) {
			printf("Child process gets SIGALRM signal\n");
			printf("Child process is alarmed\n");
		}
		else if (WTERMSIG(status) == 15) {
			printf("Child process gets SIGTERM signal\n");
			printf("Child process is terminated\n");
		}

		printf("CHILD EXECUTION FAILED\n");
	} 
	else if (WIFSTOPPED(status)) {
		if (WSTOPSIG(status) == 19) {
			printf("Child process gets SIGSTOP signal\n");
			printf("Child process stopped\n");	
		}

		printf("CHILD PROCESS STOPPED with value = %d\n",WSTOPSIG(status));
	} 
	else {
		printf("CHILD PROCESS CONTINUED\n");
	}
}
