object:=./main.out

$(object):main.cu user_program.cu virtual_memory.cu virtual_memory.h
	nvcc -rdc=true main.cu user_program.cu virtual_memory.cu -o $(object)

.PHONY:clean
clean:$(object)
	-rm -rf $(object)
