object:=./main.out

$(object):main.cu user_program.cu file_system.cu file_system.h
	nvcc -rdc=true main.cu user_program.cu file_system.cu -o $(object)

.PHONY:clean
clean:$(object)
	-rm -rf $(object)
