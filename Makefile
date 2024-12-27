all: clean boxfilter grayscale matmul vectoradd

query: src/query.cu
	nvcc -O3 -o bin/query src/query.cu

boxfilter: src/boxfilter/boxfilter.cu
	nvcc -O3 -o bin/boxfilter src/boxfilter/boxfilter.cu -I src/include

grayscale: src/grayscale/grayscale.cu
	nvcc -O3 -o bin/grayscale src/grayscale/grayscale.cu -I src/include

matmul: src/matmul/matmul.cu
	nvcc -O3 -o bin/matmul src/matmul/matmul.cu -I src/include

vectoradd: src/vectoradd/vectoradd.cu
	nvcc -O3 -o bin/vectoradd src/vectoradd/vectoradd.cu -I src/include

clean:
	rm -rf ./bin/*