#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"


// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2

#define DEV_NAME "mydev"        // name for alloc_chrdev_region
#define DEV_BASEMINOR 0         // baseminor for alloc_chrdev_region
#define DEV_COUNT 1             // count for alloc_chrdev_region

void *dma_buf;
static int dev_major;
static int dev_minor;
static struct cdev *dev_cdev;

// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static long drv_ioctl(struct file *, unsigned int , unsigned long );

// cdev file_operations
static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

// in and out function
// Interface with DMA buffer
void myoutc(unsigned char data,unsigned short int port);
void myouts(unsigned short data,unsigned short int port);
void myouti(unsigned int data,unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn {
    char a;
    int b;
    short c;
} *dataIn;


// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);

static int get_Kth_Prime (int base, int k);
static int is_prime(int n);
static int gcd(int a, int b);


// Input and output data from/to DMA
void myoutc(unsigned char data,unsigned short int port) {
    *(volatile unsigned char*)(dma_buf+port) = data;
}
void myouts(unsigned short data,unsigned short int port) {
    *(volatile unsigned short*)(dma_buf+port) = data;
}
void myouti(unsigned int data,unsigned short int port) {
    *(volatile unsigned int*)(dma_buf+port) = data;
}
unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf+port);
}
unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf+port);
}
unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}


static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
    	printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}
static int drv_release(struct inode* ii, struct file* ff) {
	module_put(THIS_MODULE);
    	printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement read operation for your device */
	int result = myini(DMAANSADDR);
	printk("%s:%s(): ans = %d\n", PREFIX_TITLE, __func__, result);

	put_user(result, (int *)buffer);

	// Clean the result
	myouti(0, DMAANSADDR);
	// Read complete
	myouti(1, DMARWOKADDR);
	// Result unreadable
	myouti(0, DMAREADABLEADDR);

	return 0;
}
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement write operation for your device */
	struct DataIn din;
	//printk("%s:%s(): write start\n", PREFIX_TITLE, __func__);

 	//dataIn = (struct DataIn *) buffer;
	//copy_from_user(dataIn, (struct DataIn*)buffer, sizeof(buffer));
	get_user(din.a, (char*)buffer);
	get_user(din.b, (int*)buffer+1);
	get_user(din.c, (int*)buffer+2);

	myoutc(din.a, DMAOPCODEADDR);
	myouti(din.b, DMAOPERANDBADDR);
	myouts(din.c, DMAOPERANDCADDR);
	
	int IOMode = myini(DMABLOCKADDR);
	//printk("%s:%s(): IO Mode is %d\n", PREFIX_TITLE, __func__, IOMode);
	INIT_WORK(work_routine, drv_arithmetic_routine);

	// Result unreadable ??
	//myouti(0, DMAREADABLEADDR);
	printk("%s:%s(): queue work\n", PREFIX_TITLE, __func__);
	// Decide io mode
	if(IOMode) {
		// Blocking IO
		printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
		schedule_work(work_routine);
		flush_scheduled_work();		
    	} 
	else {
		// Non-locking IO
		printk("%s,%s(): non-blocking\n",PREFIX_TITLE, __func__);
		schedule_work(work_routine);
		// set readable
   	 }
	//printk("%s:%s(): Write complete\n", PREFIX_TITLE, __func__);
	// Write complete
	//myouti(myini(DMARWOKADDR) | 1<<8, DMARWOKADDR);
	//myouti(1, DMARWOKADDR);
	return 0;
}
static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	/* Implement ioctl setting for your device */
	int data;	
	get_user(data, (int*)arg);
	switch (cmd)
	{
	case HW5_IOCSETSTUID:		
		myouti(data, DMASTUIDADDR);
		printk("%s,%s(): My STUID is = %d\n",PREFIX_TITLE, __func__,myini(DMASTUIDADDR));
		break;
	case HW5_IOCSETRWOK:
		myouti(data, DMARWOKADDR);
		if (myini(DMARWOKADDR)) {
			printk("%s,%s(): RW OK\n",PREFIX_TITLE, __func__);
		} else printk("%s,%s(): RW NOT OK!!!\n",PREFIX_TITLE, __func__);
		break;
	case HW5_IOCSETIOCOK:
		myouti(data, DMAIOCOKADDR);
		if (myini(DMAIOCOKADDR)) {
			printk("%s,%s(): ioc OK\n",PREFIX_TITLE, __func__);
		} else printk("%s,%s(): ioc NOT OK!!!\n",PREFIX_TITLE, __func__);
		break;
	case HW5_IOCSETIRQOK:
		myouti(data, DMAIRQOKADDR);
		if (myini(DMAIRQOKADDR)) {
			printk("%s,%s(): IRQ OK\n",PREFIX_TITLE, __func__);
		}
		break;
	case HW5_IOCSETBLOCK:
		myouti(data, DMABLOCKADDR);
		if (myini(DMABLOCKADDR)) {
			printk("%s,%s(): Blocking IO\n",PREFIX_TITLE, __func__);
		} else printk("%s,%s(): Non-blocking IO\n",PREFIX_TITLE, __func__);
		break;
	case HW5_IOCWAITREADABLE:
		while (!myini(DMAREADABLEADDR)) {
			msleep(100);
		}
		put_user(1,(int*)arg);
		printk("%s,%s(): wait readable 1\n",PREFIX_TITLE, __func__);
		break;		

	default:
		printk("Invalid command\n");
		break;
	}
	return 0;
}

static void drv_arithmetic_routine(struct work_struct* ws) {
	/* Implement arthemetic routine */
	char op;
	int opB, opC, ans;
	op = myinc(DMAOPCODEADDR);
	opB = myini(DMAOPERANDBADDR);
	opC = myins(DMAOPERANDCADDR);
	//printk("%s,%s(): opB = %d, opC = %d\n",PREFIX_TITLE, __func__,opB,opC);
	switch(op) {
        case '+':
            ans = opB + opC;
            break;
        case '-':
            ans = opB - opC;
            break;
        case '*':
            ans = opB * opC;
            break;
        case '/':
            ans = opB / opC;
            break;
        case 'p':
            ans = get_Kth_Prime(opB, opC);
            break;
        default:
            ans=0;
    }
	printk("%s,%s(): %d %c %d = %d\n\n", PREFIX_TITLE, __func__, opB, op, opC, ans);
	myouti(ans, DMAANSADDR);
	myouti(1, DMAREADABLEADDR);
}

static int gcd(int a, int b) {
    int r;
    while (a % b != 0)
    {
        r = a % b;
        a = b;
        b = r;
    }
    return b;    
}

static int is_prime(int n) {
    if (n == 2) return 1;
	int i;
    for (i = 2; i < n; i++) {
        if (gcd(i,n) != 1) return 0;
    }    
    return 1;
}

static int get_Kth_Prime(int base, int k) {
    int cnt = 0;
    int result; 
    while (cnt != k)
    {
        base++;
        if (is_prime(base)) {
            result = base;
            cnt++;
        }
    }
    return result;
}

static int __init init_modules(void) {
    
	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);
	dev_t dev;
	dev_cdev = cdev_alloc();

	/* Register chrdev */ 
	if (alloc_chrdev_region(&dev, DEV_BASEMINOR, DEV_COUNT, DEV_NAME) < 0) {
		printk(KERN_ALERT"Register chrdev failed!\n");
		return -1;
	} else {
		printk("%s:%s(): register chrdev(%i,%i)\n", PREFIX_TITLE, __func__, MAJOR(dev), MINOR(dev));
	}

	dev_major = MAJOR(dev);
	dev_minor = MINOR(dev);

	/* Init cdev and make it alive */
	dev_cdev->ops = &fops;
	dev_cdev->owner = THIS_MODULE;

	if (cdev_add(dev_cdev, dev, 1) < 0) {
		printk(KERN_ALERT"Add cdev failed!\n");
		return -1;
	}

	/* Allocate DMA buffer */
	dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);
	printk("%s:%s(): allocate dma buffer\n", PREFIX_TITLE, __func__);

	/* Allocate work routine */
	work_routine = kmalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);

	return 0;
}

static void __exit exit_modules(void) {

	/* Free DMA buffer when exit modules */
	kfree(dma_buf);
	printk("%s:%s(): free dma buffer\n", PREFIX_TITLE, __func__);
	/* Delete character device */
	unregister_chrdev_region(MKDEV(dev_major,dev_minor), DEV_COUNT);
	cdev_del(dev_cdev);
	printk("%s:%s(): unregister chrdev\n", PREFIX_TITLE, __func__);

	/* Free work routine */
	kfree(work_routine);

	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);
