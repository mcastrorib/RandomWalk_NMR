#ifndef MPI_NMR_H
#define MPI_NMR_H

#include <vector>
#include <string>

// include MPI API for multiprocesses
#include <mpi.h>

#include "NMR_defs.h"
#include "NMR_Simulation.h"
#include "../Walker/walker.h"
#include "../BitBlock/bitBlock.h"


using namespace std;

// mpi data typedefs
struct mpi_bitblock_properties
{ 
	int numberOfBlocks; 
	int imageRows; 
	int imageColumns; 
	int imageDepth; 
	int blockRows;
	int blockColumns;
	int blockDepth; 
};

struct mpi_nmr_T2
{
	double value[NMR_T2_SIZE];
};

class NMR_Network
{
public:
	NMR_Simulation &NMR;
	int mpi_rank;
	int mpi_processes;

	NMR_Network(NMR_Simulation &_NMR, int _mpi_rank, int _mpi_processes);
	virtual ~NMR_Network(){};
	void transfer();
	void time();

private:
	MPI_Datatype MPI_BITBLOCK_PROPERTIES;
	MPI_Datatype MPI_NMRT2;
	double transfer_time;

	void exchangeBitBlock();
	void sendBitBlockProperties();
	void receiveBitBlockProperties();
	void sendBitBlock();
	void sendBitBlockInBatches();
	void sendBatch(uint _batch);
	void receiveBitBlock();
	void receiveBitBlockInBatches();
	void receiveBatch(uint _batch);

	void exchange_NMRT2();
	void send_NMRT2();
	void receive_NMRT2();

	void notifyState(int _state);
	void sendStateNotification(int _state);
	void receiveStateNotifications(int _state);

	void printTime()
	{
		cout << "[" << this->mpi_rank << "] ~ NMR network transfer: \t" << this->transfer_time << " s" << endl;
	}
};

#endif