# Kernel module insertion

## Program 1
The basic steps of this task involve creating a child process in which a specific signal 
such as KILL, TERMINATE, and etc., will be sent to the parent process. As soon as 
the parent process receives the particular signal (after waiting for some time), it will
analyze the information and display how the child process terminates and what signal 
was raised in child process.

## Program 2
The procedures of task 2 are virtually the same as what we have done in task 1 expect 
that we attempt to perform some operations in the kernel space of the Linux operating 
system. In the kernel space, we cannot use fork(), execve() or wait() functions 
directly. Rather, we need to apply lower-level functions exported from the source code 
of the Linux operating system such as _do_fork(), do_execve(), do_wait()
and getname() through EXPORT_SYSMBOL().

## Environment
### Linux Version: Ubuntu 16.04
### Linux Kernel Version: 4.10.14

## Execution
### Program 1
In the 'program 1' directory, use the following command
```
make
```
to compile the files.
To test the program, type in
```
./program1 $TEST_CASE
```
### Program 2
To login your personal account, type in
```
sudo su
```
and then 
```
make
```
In some cases, you need use the following command for successful compilation
```
make CONFIG_STACK_VALIDATION=
```
Now, insert the kernel module and check the kernel message to see if we managed to insert the module.
```
insmod program2.ko
# Show the message printed in kernel mode
dmesg | tail -n 8
# Remove the module 
rmmod program2.ko
```

## Sample Outputs
The following picture shows a segmentation_fault signal captured and displayed in the kernel space.

![image](segment_fault.png)