#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

static struct task_struct *task;
static struct wait_opts
{
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;
	struct siginfo __user *wo_info;
	int __user *wo_stat;
	struct rusage __user *wo_rusage;
	wait_queue_t child_wait;
	int notask_error;
};

extern long _do_fork(unsigned long clone_flags,
					 unsigned long stack_start,
					 unsigned long stack_size,
					 int __user *parent_tidptr,
					 int __user *child_tidptr,
					 unsigned long tls);

extern int do_execve (struct filename *filename,
					  const char __user *const __user *__argv,
					  const char __user *const __user *__envp);

extern long do_wait(struct wait_opts *wo);

extern struct filename *getname(const char __user * filename);

int my_exec(void) {
	int result;
	const char path[] = "/opt/test";
	const char *const argv[] = {path,NULL,NULL};
	const char *const envp[] = {"HOME=/","PATH=/sbin:/user/sbin:/bin:/usr/bin",NULL};

	struct filename * my_filename = getname(path);

	result = do_execve(my_filename,argv,envp);

	if(!result) return 0;
	do_exit(result);
}

void my_wait(pid_t pid) {
	int status;
	struct wait_opts wo;
	struct pid *wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);

	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WEXITED | WUNTRACED;
	wo.wo_info = NULL;
	wo.wo_stat = (int __user*)&status;
	wo.wo_rusage = NULL;

	int a;
	a = do_wait(&wo);
	int sig = *wo.wo_stat;
	//printk("[program2] : do_wait return value is %d\n",&a);

	if (sig == 0) {
		printk("[program2] : Child process terminates normally\n");
	}
	else if (sig == 1) {
		printk("[program2] : Child process gets SIGHUP signal\n");
		printk("[program2] : Child process is hung up\n");
	}
	else if (sig == 2) {
		printk("[program2] : Child process gets SIGINT signal\n");
		printk("[program2] : Child process is interrupted\n");
	}
	else if (sig == 9) {
		printk("[program2] : Child process gets SIGKILL signal\n");
		printk("[program2] : Child process is killed\n");
	}
	else if (sig == 13) {
		printk("[program2] : Child process gets SIGPIPE signal\n");
		printk("[program2] : Child process attempts to write a pipe without a process connected to the other end\n");
	}
	else if (sig == 14) {
		printk("[program2] : Child process gets SIGALRM signal\n");
		printk("[program2] : Child process is alarmed\n");
	}
	else if (sig == 15) {
		printk("[program2] : Child process gets SIGTERM signal\n");
		printk("[program2] : Child process is terminated\n");
	}
	else if (sig == 131) {
		printk("[program2] : Child process gets SIGQUIT signal\n");
		printk("[program2] : Child process is quitted\n");
	}
	else if (sig == 132) {
		printk("[program2] : Child process gets SIGILL signal\n");
		printk("[program2] : Child process attempts to execute an illegal instruction\n");
	}
	else if (sig == 133) {
		printk("[program2] : Child process gets SIGTRAP signal\n");
		printk("[program2] : Child process is trapped\n");
	}
	else if (sig == 134) {
		printk("[program2] : Child process gets SIGABRT signal\n");
		printk("[program2] : Child process is aborted\n");
	}
	else if (sig == 135) {
		printk("[program2] : Child process gets SIGBUS signal\n");
		printk("[program2] : Child process reports a bus error\n");
	}
	else if (sig == 136) {
		printk("[program2] : Child process gets SIGFPE signal\n");
		printk("[program2] : Child process reports an error in arithmetic operation\n");
	}
	else if (sig == 139) {
		printk("[program2] : Child process gets SIGSEGV signal\n");
		printk("[program2] : Child process makes a segmentation fault\n");
	}
	else if (sig == 4991) {
		printk("[program2] : Child process gets SIGSTOP signal\n");
		printk("[program2] : Child process stopped\n");
	}
	else if (sig < 0) {
		printk("An error occurs. Please check if your path is correct\n");
	}
	
	printk("[program2] : The return signal is %d\n",sig);
	put_pid(wo_pid);
	return;
}

int my_fork(void *argc){
	//set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	/* fork a process using do_fork */
	pid_t pid = _do_fork(SIGCHLD,(unsigned long)&my_exec,0,NULL,NULL,0);
	printk("[program2] : The child process has pid = %d\n",pid);
	printk("[program2] : This is the parent process, pid = %d\n",(int)current->pid);
	/* execute a test program in child process */		
	
	/* wait until child process terminates */
	my_wait(pid);
	return 0;
}

static int __init program2_init(void){
	printk("[program2] : Module_init\n");	
	/* create a kernel thread to run my_fork */
	//printk("[program2] : Module_init create kthread starts");
	task = kthread_create(&my_fork,NULL,"MyThread");
	if (!IS_ERR(task)) {
		printk("[program2] : Module_init kthread starts\n");
		wake_up_process(task);
	}
	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);