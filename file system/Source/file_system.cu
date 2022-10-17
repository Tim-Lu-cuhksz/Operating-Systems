#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NAME_START 0
#define TIME_START 20
#define SIZE_START 24
#define BLOCK_START 26
#define LENGTH_START 28

__device__ __managed__ u32 gtime = 0;

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE; //4096
  fs->FCB_SIZE = FCB_SIZE; //32 bytes
  fs->FCB_ENTRIES = FCB_ENTRIES; //1024
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE; // 32 bytes
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE; // 20 bytes
  fs->MAX_FILE_NUM = MAX_FILE_NUM; //1024
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE; //1024 bytes
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

	// We need to initialize super block and FCB to 0
  for (int i = 0; i < fs->FILE_BASE_ADDRESS; i++) {
	  fs->volume[i] = 0x0; 
	}
}

// Translate block index to real address
// Assume the starting block number is 0 rather than 1;
__device__ int block_to_file_addr(FileSystem *fs, u32 file_start_block) {
	return fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE * file_start_block;
}

// Translate fcb block index fcb starting address
__device__ int block_to_fcb_addr(FileSystem* fs, u32 file_start_block) {
	return fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * file_start_block;
}

// Given starting address and character, check if equal
__device__ bool FCB_name_check(FileSystem* fs, char* s, u32 addr)
{
	int i = 0;
	do {
		if (fs->volume[addr + i] != s[i]) return false;
		//i++;
	} while (s[i++] != '\0');
	return true;
}

__device__ void FCB_get_name(FileSystem* fs, char* output, u32 addr)
{
	int i = 0;
	while (fs->volume[addr + i] != '\0') {
		output[i] = fs->volume[addr + i];
		i++;
	}
	output[i] = '\0';
	return;
}

__device__ int update_free_space(FileSystem* fs, u32 byte_index, 
								u32 bit_index, u32 count) {
	for (int i = 0; i < count; i++) {
		if (byte_index < 0) {
			printf("update_free_space: byte index overflow\n");
			return -1;
		}
		// Update the bit to 1 -> used
		fs->volume[byte_index] |= 1 << bit_index;
		if (bit_index == 7) {
			bit_index = 0;
			byte_index--;
		} else bit_index++;
	}
}

// Contiguous block allocation
// Return the index of the first block
// Not find: return -1
__device__ int super_find_empty(FileSystem* fs, u32 size) {
	int quotient = size / fs->STORAGE_BLOCK_SIZE;
	int remainder = size % fs->STORAGE_BLOCK_SIZE;
	int number_of_blocks;
	if (size == 0) number_of_blocks = 1;
	else number_of_blocks = (remainder == 0) ? quotient : quotient + 1;
	//printf("Number of blocks %d\n", number_of_blocks);
	//int flag = 0;
	int cnt = 0, byte_index, bit_index;
	for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
		for (int j = 7; j >= 0; j--) {
			// if the bit is 0 -> unused
			if (!(fs->volume[i] >> j & 0b01)) {
				cnt++;
			}
			else {
				cnt = 0;
			}
			if (cnt == number_of_blocks) {
				/* Update the old free space */
				update_free_space(fs, i, j, cnt);
				return 8 * (i + 1) - j - cnt;
			}
		}
	}
	return -1;
}

// Find an empty space in FCB table
// If find, return the address of the FCB
__device__ u32 FCB_find_empty(FileSystem* fs) {
	u32 current_addr;
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		current_addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i;
		// If it is unused
		if (!(fs->volume[current_addr + fs->FCB_SIZE - 1] & 0b01)) {
			return current_addr;
		}
	}
	return 0;
}

// Find if the file has already existed
// If yes: return fcb address
// Else: return 0
__device__ u32 FCB_find_file(FileSystem* fs, char* s)
{
	//u32 fp;	// Block number + number of blocks
	//printf("%s\n",s);
		
	u32 current_addr; // Current address of FCB
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		current_addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i;
		// If it is used
		if (fs->volume[current_addr + fs->FCB_SIZE - 1] & 0b01) {
			// if the name is matched
			if (FCB_name_check(fs, s, current_addr)) {
				//printf(" address: %d\n", current_addr);
				return current_addr;
			}
		}
	}
	return 0;
}

__device__ u32 FCB_update_time(FileSystem* fs, u32 fcb_addr, u32 time) {
	// Update time 4 bytes: big endian
	// LSB stores in higher address
	for (int i = 0; i < 4; i++) {
		fs->volume[fcb_addr + TIME_START + 3 - i] &= 0x0;
		fs->volume[fcb_addr + TIME_START + 3 - i] |= (time >> 8 * i) & 0xFF;
	}
}

__device__ u32 FCB_update_size(FileSystem* fs, u32 fcb_addr, u32 size) {
	// Update size 2 bytes: big endian
	for (int i = 0; i < 2; i++) {
		fs->volume[fcb_addr + SIZE_START + 1 - i] &= 0x0;
		fs->volume[fcb_addr + SIZE_START + 1 - i] |= (size >> 8 * i) & 0xFF;
	}
}

__device__ u32 FCB_update_block(FileSystem* fs, u32 fcb_addr, u32 block) {
	// Update block number 2 bytes
	for (int i = 0; i < 2; i++) {
		fs->volume[fcb_addr + BLOCK_START + 1 - i] &= 0x0;
		fs->volume[fcb_addr + BLOCK_START + 1 - i] |=
			(block >> 8 * i) & 0xFF;
	}
}

__device__ u32 FCB_update_length(FileSystem* fs, u32 fcb_addr, u32 length) {
	// Update length 2 bytes
	for (int i = 0; i < 2; i++) {
		fs->volume[fcb_addr + LENGTH_START + 1 - i] &= 0x0;
		fs->volume[fcb_addr + LENGTH_START + 1 - i] |= (length >> 8 * i) & 0xFF;
	}
}

__device__ u32 FCB_update(FileSystem* fs, u32 fcb_addr, char* name, u32 time, 
						u32 size, u32 start_block_num, u32 length) {
	// Update name
	int i = 0;
	while (name[i] != '\0')
	{
		fs->volume[fcb_addr + i] = name[i];
		i++;
	}
	fs->volume[fcb_addr + i] = '\0';

	FCB_update_time(fs, fcb_addr, time);
	FCB_update_size(fs, fcb_addr, size);
	FCB_update_block(fs, fcb_addr, start_block_num);
	FCB_update_length(fs, fcb_addr, length);
	// Update used bit
	fs->volume[fcb_addr + fs->FCB_SIZE - 1] |= 0b01;
}

__device__ u32 total_empty_size(FileSystem* fs) {
	u32 cnt;
	for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
		for (int j = 7; j >= 0; j--) {
			// Unused
			if (!(fs->volume[i] >> j & 0b01)) cnt++;
		}
	}
	return cnt * fs->STORAGE_BLOCK_SIZE;
}

__device__ u32 getDate(FileSystem* fs, u32 block_addr) {
		u32 time = fs->volume[block_addr + TIME_START + 3] |
			fs->volume[block_addr + TIME_START + 2] << 8;
		time |= fs->volume[block_addr + TIME_START + 1] << 16;
		return time;
}

__device__ u32 getSize(FileSystem* fs, u32 block_addr) {
		u32 size = fs->volume[block_addr + SIZE_START + 1] |
		fs->volume[block_addr + SIZE_START] << 8;
		return size;
}

__device__ u32 getBlockNum(FileSystem* fs, u32 block_addr) {
	u32 idx = fs->volume[block_addr + BLOCK_START + 1] |
		fs->volume[block_addr + BLOCK_START] << 8;
	return idx;
}

__device__ void swap(FileSystem* fs, u32 block_addr1, u32 block_addr2) {
	for (int i = 0; i < fs->FCB_SIZE; i++) {
		uchar tmp = fs->volume[block_addr1 + i];
		fs->volume[block_addr1 + i] = fs->volume[block_addr2 + i];
		fs->volume[block_addr2 + i] = tmp;
	}
}

// low: Lower block idx
// high: Higher block idx
__device__ int partition(FileSystem* fs, int low, int high, int flag) {
	// Select the last element as our pivot
	u32 key, comp;
	int i = low - 1;
	for (int j = low; j < high; j++) {
		if (flag == 1) {
			key = getBlockNum(fs, block_to_fcb_addr(fs, high));
			comp = getBlockNum(fs, block_to_fcb_addr(fs, j));
			if (comp <= key) {
				i++;
				// Exchange array[i] and array[j]
				swap(fs, block_to_fcb_addr(fs, i), block_to_fcb_addr(fs, j));
			}
		}
		else if (flag == 0) {
			// Compare size
			key = getSize(fs, block_to_fcb_addr(fs, high));
			comp = getSize(fs, block_to_fcb_addr(fs, j));
			if (comp == key) { // First modified first print
				key = getDate(fs, block_to_fcb_addr(fs, high));
				comp = getDate(fs, block_to_fcb_addr(fs, j));
				if (comp > key) {
					i++;
					// Exchange array[i] and array[j]
					swap(fs, block_to_fcb_addr(fs, i), block_to_fcb_addr(fs, j));
				}
			}
			else if (comp > key) {
				i++;
				// Exchange array[i] and array[j]
				swap(fs, block_to_fcb_addr(fs, i), block_to_fcb_addr(fs, j));
			}
		}
		else {
			//Compare date
			key = getDate(fs, block_to_fcb_addr(fs, high));
			comp = getDate(fs, block_to_fcb_addr(fs, j));
			if (comp > key) {
				i++;
				// Exchange array[i] and array[j]
				swap(fs, block_to_fcb_addr(fs, i), block_to_fcb_addr(fs, j));
			}
		}
	}
	// Exchange array[i+1] and array[high]
	swap(fs, block_to_fcb_addr(fs, i + 1), block_to_fcb_addr(fs, high));
	return i + 1;
}


// Flag: 1->increase block number; 0->decrease size; 2->decrease time
__device__ void quickSort(FileSystem* fs, int low, int high, int flag) {
	if (low < high) {
		int q = partition(fs, low, high, flag);
		quickSort(fs, low, q - 1, flag);
		quickSort(fs, q + 1, high, flag);
	}
}

__device__ void insertionSort(FileSystem* fs, int low, int high, int flag) {
	int key, j;
	if (flag == 0) {
		for (int i = low + 1; i <= high; i++) {
			//key = array[i];
			key = getBlockNum(fs, block_to_fcb_addr(fs, i));
			j = i - 1;
			while (j >= low && getBlockNum(fs,block_to_fcb_addr(fs, j)) > key) {
				swap(fs, block_to_fcb_addr(fs, j + 1), block_to_fcb_addr(fs, j));
				j--;
			}
		}
	}
	else if (flag == 1) {
		for (int i = high - 1; i >= low; i--) {
			//key = array[i];
			key = getSize(fs, block_to_fcb_addr(fs, i));
			j = i + 1;
			//printf("key: %d\n", key);
			//printf("j: %d\n", getSize(fs, block_to_fcb_addr(fs, j)));
			while (j <= high && getSize(fs, block_to_fcb_addr(fs, j)) > key) {
				//printf("Enter while loop.\n"); break; break;
				swap(fs, block_to_fcb_addr(fs, j - 1), block_to_fcb_addr(fs, j));
				j++;
				
			while (j <= high && getSize(fs, block_to_fcb_addr(fs, j)) == key &&
					getDate(fs,block_to_fcb_addr(fs, j))) {
				getDate(fs, block_to_fcb_addr(fs, i));
				swap(fs, block_to_fcb_addr(fs, j - 1), block_to_fcb_addr(fs, j));
				j++;					
				}
			}
		}
	}
	else {
		for (int i = high - 1; i >= low; i--) {
			//key = array[i];
			key = getDate(fs, block_to_fcb_addr(fs, i));
			j = i + 1;
			//printf("key: %d\n", key);
			//printf("j: %d\n", getDate(fs, block_to_fcb_addr(fs, j)));
			while (j <= high && getDate(fs, block_to_fcb_addr(fs, j)) > key) {
				//printf("Enter while loop.\n"); break; break;
				swap(fs, block_to_fcb_addr(fs, j-1), block_to_fcb_addr(fs, j));
				j++;
			}
		}
	}
}




// If size not equal to zero
__device__ u32 getLength(FileSystem* fs, u32 size) {
	int quotient = size / fs->STORAGE_BLOCK_SIZE;
	int remainder = size % fs->STORAGE_BLOCK_SIZE;
	return (remainder == 0) ? quotient : quotient + 1;
}



__device__ void mask(FileSystem* fs, u32 block_num, u32 length) {
	u32 byte_index = block_num / 8;
	u32 bit_index = 7 - block_num % 8;
	for (int i = 0; i < length; i++) {
		fs->volume[byte_index] &= ~(1 << bit_index);
		if (bit_index == 0) {
			bit_index = 7;
			byte_index++;
		}
		else bit_index--;
	}
}

__device__ void unmask(FileSystem* fs, u32 block_num, u32 length) {
	u32 byte_index = block_num / 8;
	u32 bit_index = 7 - block_num % 8;
	for (int i = 0; i < length; i++) {
		fs->volume[byte_index] |= 1 << bit_index;
		if (bit_index == 0) {
			bit_index = 7;
			byte_index++;
		}
		else bit_index--;
	}
}

__device__ void fcb_gap_eliminate(FileSystem* fs) {
	int gap_idx = -1, fcb_idx = -1;
	u32 gap_addr, fcb_addr;
	// Loop through 1024 entries
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		// If not used -> gap
		// Try to fill in the gap if there are blocks after it
		gap_addr = block_to_fcb_addr(fs, i);
		if (!(fs->volume[gap_addr + fs->FCB_SIZE - 1] & 0b01)) {
			//if (gap_idx == -1) gap_idx = i;
			for (int j = i + 1; j < fs->FCB_ENTRIES; j++) {
				// We find a block and then fill it with the gap
				fcb_addr = block_to_fcb_addr(fs, j);
				if (fs->volume[fcb_addr + fs->FCB_SIZE - 1] & 0b01) {
					// Copy to blank fcb
					for (int k = 0; k < fs->FCB_SIZE; k++) {
						fs->volume[gap_addr + k] = fs->volume[fcb_addr + k];
					}
					// Set the original fcb unused
					fs->volume[fcb_addr + fs->FCB_SIZE - 1] &= 0b11111110;
					break;
				}
			}
		}
	}
}

// Return the new allocated block number
__device__ u32 reallocation(FileSystem* fs, u32 new_size, u32 FCB_addr, u32 FCB_length) {
	// Reconfigure the free space
	for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
		fs->volume[i] = 0x0; // 0 -> unused
	}

	// Remove the old file and its fcb
	fs->volume[FCB_addr + fs->FCB_SIZE - 1] &= 0b11111110;
	for (int i = 0; i < fs->STORAGE_BLOCK_SIZE * FCB_length; i++) {
		fs->volume[FCB_addr + i] = 0x0;
	}

	// Reallocate the rest of the file
	int cnt = 0;
	u32 current_addr; // Current address of FCB
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		current_addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i;
		// If it is used
		if (fs->volume[current_addr + fs->FCB_SIZE - 1] & 0b01) {
			// Reallocate the rest of the files
			if (current_addr != FCB_addr) {
				cnt++;
			}
		}
	}
	// gap_eliminate
	fcb_gap_eliminate(fs);
	// Sort the block in increasing order
	//quickSort(fs, 0, cnt-1, 1);
	insertionSort(fs, 0, cnt - 1, 0);

	int old_file_block, new_file_block, new_length;
	u32 size, fcb_addr, old_file_addr, new_file_addr;
	for (int i = 0; i < cnt; i++) {
		fcb_addr = block_to_fcb_addr(fs, i);
		// Size: measured in bytes
		size = getSize(fs, fcb_addr);

		new_length = (size != 0) ? getLength(fs, size) : 1;

		old_file_block = getBlockNum(fs, fcb_addr);
		old_file_addr = block_to_file_addr(fs, old_file_block);

		new_file_block = super_find_empty(fs, size);
		new_file_addr = block_to_file_addr(fs, new_file_block);
		// Change the contents to new address
		for (int i = 0; i < size; i++) {
			fs->volume[new_file_addr + i] = fs->volume[old_file_addr + i];
		}
		// Clean
		for (int i = size; i < fs->STORAGE_BLOCK_SIZE * new_length; i++) {
			fs->volume[new_file_addr + i] = 0x0;
		}
		// Only need to update block because other attributes do not change
		FCB_update_block(fs, fcb_addr, new_file_block);
	}

	// Allocate to the end, update and return
	// Find new space
	return super_find_empty(fs,new_size);
}


__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	// Find the file in FCB table and return its address
	u32 fp = FCB_find_file(fs, s);
	if (fp != 0) return fp;
	if (op == G_READ) { // Invalid read
		printf("Read file missing!\n");
		return 0;
	}
	else { // Create a new file with size 0 byte
		// We find a space and update the free space
		int block_num = super_find_empty(fs, 0);
		u32 FCB_addr = FCB_find_empty(fs);
		if (FCB_addr != 0) {
			if (block_num != -1) {
				// Update FCB
				FCB_update(fs, FCB_addr, s, gtime, 0, block_num, 1);
				// Return a new fp
				return FCB_addr;
			}
			else {
				printf("No free space.");
				return 0;
			}
		}
		else {
			printf("FCB full!");
			return 0;
		}		
	}
	return 0;
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	if (fp == 0) {
		printf("fs_read: Fail to open the file.\n");
		return;
	}
	u32 block_num = fs->volume[fp + BLOCK_START + 1];
	block_num |= fs->volume[fp + BLOCK_START] << 8;
	u32 length = fs->volume[fp + LENGTH_START + 1];
	length |= fs->volume[fp + LENGTH_START] << 8;

	//printf("fs_read: starts.\n");

	if (size > length * fs->STORAGE_BLOCK_SIZE) {
		printf("fs_read: read size larger than file size.\n");
		return;
	}
	u32 addr = block_to_file_addr(fs, block_num);
	for (int i = 0; i < size; i++) {
		output[i] = fs->volume[addr + i];
	}
	//printf("fs_read: finishes.\n");
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	if (fp == 0) {
		printf("fs_write: Fail to open the file.\n");
		return;
	}
	gtime++; // Modify symbol

	u32 block_num = fs->volume[fp + BLOCK_START + 1];
	block_num |= fs->volume[fp + BLOCK_START] << 8;
	u32 length = fs->volume[fp + LENGTH_START + 1];
	length |= fs->volume[fp + LENGTH_START] << 8;

	if (size > length * fs->STORAGE_BLOCK_SIZE) {
		// Try to reallocate
		if (total_empty_size(fs) >= size) {
			u32 addr;
			u32 new_length = getLength(fs, size);

			// Mask: mask the current file in case
			// The adjacent spcae is large enough for allocation
			mask(fs, block_num, length);
			u32 new_block = super_find_empty(fs, size);
			if (new_block != -1) { // Free space automatically updated
				addr = block_to_file_addr(fs,new_block);
				for (int i = 0; i < size; i++) {
					fs->volume[addr + i] = input[i];
				}
				// Clean the other space
				for (int i=size; i < fs->STORAGE_BLOCK_SIZE * new_length; i++){
					fs->volume[addr + i] = 0x0;
				}
				// Update				
				FCB_update_block(fs, fp, new_block);
				FCB_update_length(fs, fp, new_length);
				FCB_update_size(fs, fp, size);
				FCB_update_time(fs, fp, gtime);
			}
			else {
				// Reallocate
				unmask(fs, block_num, length);
				// Old length -> clean old files
				char old_name[20];
				FCB_get_name(fs, old_name, fp);
				u32 block = reallocation(fs, size, fp, length);
				u32 new_fcb_addr = FCB_find_empty(fs);
				u32 new_file_addr = block_to_file_addr(fs, block);
				//addr = block_to_file_addr(fs, block);
				for (int i = 0; i < size; i++) {
					fs->volume[new_file_addr + i] = input[i];
				}
				for (int i = size; i < fs->STORAGE_BLOCK_SIZE * new_length; i++) {
					fs->volume[new_file_addr + i] = 0x0;
				}
				// Update				
				FCB_update(fs, new_fcb_addr, old_name, gtime, size, block, new_length);
			}
			
		}
		else {
			printf("fs_write: Not enough space to write in a file.\n");
			return 0;
		}
	}
	else { // clean + write
		u32 addr = block_to_file_addr(fs, block_num);
		// clean
		for (int i = 0; i < fs->STORAGE_BLOCK_SIZE * length; i++) {
			fs->volume[addr + i] = 0x0;
		}
		// Write
		for (int i = 0; i < size; i++) {
			fs->volume[addr + i] = input[i];
		}
		FCB_update_size(fs, fp, size);
		FCB_update_time(fs, fp, gtime);
	}
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	u32 current_addr, size, FCB_num, time;
	int cnt = 0;
	
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		current_addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i;
		//printf("current addr: %d\n", current_addr);
		// If it is used
		if (fs->volume[current_addr + fs->FCB_SIZE - 1] & 0b01) {
			cnt++;
		}
	}
	//printf("cnt = %d\n", cnt);
	if (cnt == 0) {
		printf("No file in the directory.\n");
		return;
	}
	if (op == LS_D) { // Order by time
		fcb_gap_eliminate(fs);
		//quickSort(fs, 0, cnt - 1, 2);
		insertionSort(fs, 0, cnt - 1, 2);

		printf("===sort by modified time===\n");
		char name[20];
		for (int i = 0; i < cnt; i++) {
			current_addr = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
			//printf("current address: %x\n", current_addr);
			FCB_get_name(fs, name, current_addr);
			printf("%s\n", name);
		}
	}
	else { // Order by size
		// Sort the block in decreasing order
		fcb_gap_eliminate(fs);
		//quickSort(fs, 0, cnt - 1,0);
		insertionSort(fs, 0, cnt - 1, 1);
		printf("===sort by file size===\n");
		char name[20];		
		for (int i = 0; i < cnt; i++) {
			current_addr = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
			FCB_get_name(fs, name, current_addr);
			printf("%s %d\n",name, getSize(fs, current_addr));
		}
	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */	
	if (op == RM) {
		//FCB unused
		u32 fp = FCB_find_file(fs, s);
		if (fp == 0) {
			printf("rm: try to remove a non-existing file.");
			return;
		}
		fs->volume[fp + fs->FCB_SIZE - 1] &= 0b11111110;
		//Free space unmask
		u32 block_num = getBlockNum(fs, fp);
		u32 length = getLength(fs, fp);
		unmask(fs, block_num, length);
		u32 file_addr = block_to_file_addr(fs,block_num);
		// Clean file
		for (int i = 0; i < fs->STORAGE_BLOCK_SIZE * length; i++) {
			fs->volume[file_addr + i] = 0x0;
		}
	}
}
