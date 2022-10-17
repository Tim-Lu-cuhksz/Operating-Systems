#include "virtual_memory.h"
#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i; // Virtual page number
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer; // physical memory RAM
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}


__device__ u32 getLRUIndex(VirtualMemory *vm) {   
    u32 index = 0;
    u32 maximum = vm->invert_page_table[index] & 0x3FFFFFFF; // get rid of first two bits
    for (int i = 1; i < vm->PAGE_ENTRIES; i++) {
        if ((vm->invert_page_table[i] & 0x3FFFFFFF) > maximum) {
            maximum = vm->invert_page_table[i];
            index = i;
        }
    }
    return index;
}

/* 
    disk_frame_num (frame that will be replaced in) <-> virtual page number
    RAM_frame_num <-> page table index
*/
__device__ void replace(VirtualMemory* vm, u32 disk_frame_num, u32 RAM_frame_num) {
    u32 disk_frame_start = (vm->PAGESIZE) * disk_frame_num;

    u32 RAM_frame_start = (vm->PAGESIZE) * RAM_frame_num;
    u32 RAM_frame_end = RAM_frame_start + (vm->PAGESIZE) - 1;
    // Dirty bits operation: Check if the dirty bit is one  
    if (vm->invert_page_table[RAM_frame_num] >> 30 & 0x01) {
        // The disk frame to be written in
        u32 old_disk_frame = vm->invert_page_table[RAM_frame_num + vm->PAGE_ENTRIES];
        u32 old_disk_start = (vm->PAGESIZE) * old_disk_frame;
        if (old_disk_frame >= 2 << 12) {
            printf("Disk frame index: %d\n", old_disk_frame);
            printf("Storage overflow.\n");
            return;
        }
        // RAM -> Disk
        for (int i = RAM_frame_start; i <= RAM_frame_end; i++) {
            vm->storage[old_disk_start] = vm->buffer[i];
            old_disk_start++;
        }
    }
    // Disk -> RAM
    if (disk_frame_num >= 2 << 12) {
        printf("Disk frame index: %d\n", disk_frame_num);
        printf("Storage overflow.\n");
        return;
    }
    for (int i = RAM_frame_start; i <= RAM_frame_end; i++) {
        vm->buffer[i] = vm->storage[disk_frame_start];
        disk_frame_start++;
    } 
    // Page table reconfiguration
    vm->invert_page_table[RAM_frame_num] &= 0x0;
    vm->invert_page_table[RAM_frame_num + vm->PAGE_ENTRIES] = disk_frame_num;
    return;
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
    uchar result; int flag = 0;
    u32 offset = addr & 0b11111;
    // vpn: virtual page number
    u32 vpn = addr >> 5;
    u32 phy_addr;

    int invalid_index = -1; // For replacement
    int pt_index;

    for (pt_index = 0; pt_index < vm->PAGE_ENTRIES; pt_index++) {
        // Check if the physical memory is invalid
        if ((vm->invert_page_table[pt_index] >> 31) & 0x01) {
            if (invalid_index == -1) invalid_index = pt_index;
        }
        else { // If valid, we check the equality of vpn and the pt value
            if (vpn == vm->invert_page_table[pt_index + vm->PAGE_ENTRIES]) {
                phy_addr = (pt_index << 5) | offset;
                result = vm->buffer[phy_addr];
                // After using, we reset to 0, which means recently used
                vm->invert_page_table[pt_index] &= 0x40000000;
                flag = 1;
                break;
            }          
        }
    }
    if (flag) {
        for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
            if (i != pt_index) vm->invert_page_table[i] += 1;
        }
        return result;
    }

    // If we cannot find the data in RAM, we will perform replacement
    u32 replace_pt_index;
    *(vm->pagefault_num_ptr) += 1;

    // RAM is full 
    if (invalid_index == -1) {     
        replace_pt_index = getLRUIndex(vm); // Page index for replacement
        replace(vm, vpn, replace_pt_index);
        phy_addr = (replace_pt_index << 5) | offset;
        result = vm->buffer[phy_addr];

        for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
            if (i != replace_pt_index) vm->invert_page_table[i] += 1;
        }
    } else { // RAM not full
        replace(vm, vpn, invalid_index);
        phy_addr = (invalid_index << 5) | offset;
        result = vm->buffer[phy_addr];

        for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
            if (i != invalid_index) vm->invert_page_table[i] += 1;
        }
    }
  return result;
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
    u32 offset = addr & 0b11111; int flag = 0;
    // vpn: virtual page number
    u32 vpn = addr >> 5;
    u32 phy_addr;

    int invalid_index = -1; // For replacement
    int pt_index;

    for (pt_index = 0; pt_index < vm->PAGE_ENTRIES; pt_index++) {
        // Check if the physical memory is invalid
        if ((vm->invert_page_table[pt_index] >> 31) & 0x01) {
            if (invalid_index == -1) invalid_index = pt_index;
        }
        else { // If valid, we check the equality of vpn and the pt value
            if (vpn == vm->invert_page_table[pt_index + vm->PAGE_ENTRIES]) {
                phy_addr = (pt_index << 5) | offset;
                vm->buffer[phy_addr] = value;
                vm->invert_page_table[pt_index] |= 0x40000000; // bit[30] = 1 (dirty)
                vm->invert_page_table[pt_index] &= 0x40000000;
                flag = 1;
                break;
            }
        }
    }
    if (flag) {
        for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
            if (i != pt_index) vm->invert_page_table[i] += 1;                  
        }      
        return;
    }

    // Page fault
    *(vm->pagefault_num_ptr) += 1;
    u32 replace_pt_index;
    // RAM not full
    if (invalid_index != -1) {
        phy_addr = (invalid_index << 5) | offset;
        vm->buffer[phy_addr] = value;
        vm->invert_page_table[invalid_index] &= 0x0;
        vm->invert_page_table[invalid_index] |= 0x40000000; // dirty
        vm->invert_page_table[invalid_index + vm->PAGE_ENTRIES] = vpn;

        for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
            if (i != invalid_index) vm->invert_page_table[i] += 1;
        }
    }
    else { // RAM is full
        replace_pt_index = getLRUIndex(vm); // Page index for replacement     
        replace(vm, vpn, replace_pt_index);
        phy_addr = (replace_pt_index << 5) | offset;
        vm->buffer[phy_addr] = value;
        vm->invert_page_table[replace_pt_index] |= 0x40000000; // dirty
        for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
            if (i != replace_pt_index) vm->invert_page_table[i] += 1;
        }
    }
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
    for (int i = offset; i < input_size;  i++) {
        results[i] = vm_read(vm, i);
    }
}

