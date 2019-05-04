#include "coordinates_dataset.hh"
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

CoordinatesDataset::CoordinatesDataset(size_t batch_size, size_t number_of_batches) :
	batch_size(batch_size), number_of_batches(number_of_batches)
{
	for (int i = 0; i < number_of_batches; i++) {
		targets.push_back(Matrix(Shape(batch_size, 1)));
		targets[i].allocateMemory();
		targets[i].copyHostToDevice();
	}
}

int CoordinatesDataset::getNumOfBatches() {
	return number_of_batches;
}

std::vector<Matrix>& CoordinatesDataset::getBatches() {
	return batches;
}

std::vector<Matrix>& CoordinatesDataset::getTargets() {
	return targets;
}

Matrix CoordinatesDataset::getBatch(char* filename) {
    Matrix dummy_batch(Shape(batch_size, 2));
    dummy_batch.allocateMemory();

	int fd=open(filename,O_RDONLY);
    pread(fd, dummy_batch.data_host.get(), batch_size*2*sizeof(float), 0);
    dummy_batch.copyHostToDevice();

	close(fd);
    return dummy_batch;

}
