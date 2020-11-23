#include <iostream>
#include <unistd.h>
#include <string>
#include <vector>
#include <limits>
#include <random>
#include <algorithm>

// include MPI API for multiprocesses
#include <mpi.h>
#include <omp.h>

#include "NMR_Network.h"
#include "NMR_Simulation.h"
#include "../Walker/walker.h"
#include "../BitBlock/bitBlock.h"

using namespace std;

NMR_Network::NMR_Network(NMR_Simulation &_NMR, 
						 int _mpi_rank, 
						 int _mpi_processes) : NMR(_NMR),
								 		 	   mpi_rank(_mpi_rank),
								 		 	   mpi_processes(_mpi_processes),
								 		 	   transfer_time(0.0)
{
	// commit mpi static datatypes
	MPI_Type_contiguous(this->NMR.rwNMR_config.getBitBlockPropertiesSize(), MPI_INT, &MPI_BITBLOCK_PROPERTIES);
	MPI_Type_commit(&MPI_BITBLOCK_PROPERTIES);

	MPI_Type_contiguous(this->NMR.rwNMR_config.getNMRT2Size(), MPI_DOUBLE, &MPI_NMRT2);
	MPI_Type_commit(&MPI_NMRT2);
}

void NMR_Network::transfer()
{
	double time = omp_get_wtime();

	cout << "[" << this->mpi_rank << "]" << " ~ starting mpi communication..." << endl;
	(*this).notifyState(this->NMR.rwNMR_config.getStartTag());   

	cout << "[" << this->mpi_rank << "]" << " ~ sending/receiving bitblocks..." << endl;
	(*this).notifyState(this->NMR.rwNMR_config.getBitBlockTag());
	(*this).exchangeBitBlock();

	cout << "[" << this->mpi_rank << "]" << " ~ sending/receiving NMR-T2..." << endl;
	(*this).notifyState(this->NMR.rwNMR_config.getT2Tag()); 
	(*this).exchange_NMRT2();

	cout << "[" << this->mpi_rank << "]" << " ~ data exchange is finished..." << endl;
	(*this).notifyState(this->NMR.rwNMR_config.getEndTag());
	
	cout << "[" << this->mpi_rank << "]" << " ~ closing mpi communication..." << endl;
	this->transfer_time += omp_get_wtime() - time;
	(*this).time();
}

void NMR_Network::exchangeBitBlock()
{
	// send properties
	if(mpi_rank == 0)
	{
		sendBitBlockProperties();
	} else
	{
		receiveBitBlockProperties();
	}

	// send blocks
	if(mpi_rank == 0)
	{
		// (*this).sendBitBlock();
		(*this).sendBitBlockInBatches();
	} else
	{
		// (*this).receiveBitBlock();
		(*this).receiveBitBlockInBatches();
	}
}

void NMR_Network::sendBitBlockProperties()
{
	// wrap 'BitBlock' properties data
	mpi_bitblock_properties properties;
	properties.numberOfBlocks = NMR.bitBlock.numberOfBlocks; 
	properties.imageRows = NMR.bitBlock.imageRows; 
	properties.imageColumns = NMR.bitBlock.imageColumns; 
	properties.imageDepth = NMR.bitBlock.imageDepth; 
	properties.blockRows = NMR.bitBlock.blockRows;
	properties.blockColumns = NMR.bitBlock.blockColumns;
	properties.blockDepth = NMR.bitBlock.blockDepth;

	for(uint proc = 0; proc < this->mpi_processes; proc++)
    {
        if(proc != this->mpi_rank)
        {
            // MPI send
            int destination = proc;
            int tag = 1; 
            int data_size = 1;
            MPI_Send(&properties,
                     data_size,
                     MPI_BITBLOCK_PROPERTIES,
                     destination,
                     tag,
                     MPI_COMM_WORLD);
		}
    }
}

void NMR_Network::receiveBitBlockProperties()
{
	mpi_bitblock_properties properties;
	
	// MPI receive          
	int source = 0; // master thread
	int tag = 1; 
	int data_size = 1;
	MPI_Status status;
	MPI_Recv(&properties,
		     data_size,
		     MPI_BITBLOCK_PROPERTIES,
		     source,
		     tag,
		     MPI_COMM_WORLD,
		     &status);

	// unwrap 'BitBlock' properties data
	NMR.bitBlock.numberOfBlocks = properties.numberOfBlocks; 
	NMR.bitBlock.imageRows = properties.imageRows; 
	NMR.bitBlock.imageColumns = properties.imageColumns; 
	NMR.bitBlock.imageDepth = properties.imageDepth; 
	NMR.bitBlock.blockRows = properties.blockRows;
	NMR.bitBlock.blockColumns = properties.blockColumns;
	NMR.bitBlock.blockDepth = properties.blockDepth; 
}

void NMR_Network::sendBitBlock()
{
	uint batches = (uint) ceil(this->NMR.bitBlock.numberOfBlocks / (double) this->NMR.rwNMR_config.getBitBlockBatchesSize());	
	cout << "[" << this->mpi_rank << "]" << " ~ batches = " << batches << endl;


	// wrap 'BitBlocks' data
	uint64_t data[this->NMR.bitBlock.numberOfBlocks]; 
	for(uint i = 0; i < this->NMR.bitBlock.numberOfBlocks; i++)
	{
		data[i] = NMR.bitBlock.blocks[i];
	}

	for(uint proc = 0; proc < this->mpi_processes; proc++)
    {
        if(proc != this->mpi_rank)
        {
            // MPI send
            int destination = proc;
            int tag = 2; 
            int data_size = this->NMR.bitBlock.numberOfBlocks;
            MPI_Send(&data,
                     data_size,
                     MPI_UINT64_T,
                     destination,
                     tag,
                     MPI_COMM_WORLD);
		}
    }
}

void NMR_Network::sendBitBlockInBatches()
{
	// set number of batches
	uint batches = (uint) ceil(this->NMR.bitBlock.numberOfBlocks / (double) this->NMR.rwNMR_config.getBitBlockBatchesSize());	

	for(uint batch = 0; batch < batches; batch++)
	{
		cout << "[" << this->mpi_rank << "]" << " ~ sending batch " << batch+1 << "/" << batches << endl;
		(*this).sendBatch(batch);	
	}
}

void NMR_Network::sendBatch(uint _batch)
{
	// set offset in data access
	uint batch_offset = _batch * this->NMR.rwNMR_config.getBitBlockBatchesSize();

	// wrap 'BitBlocks' data
	uint64_t data[this->NMR.rwNMR_config.getBitBlockBatchesSize()];
	uint dataIndex; 
	for(uint id = 0; id < this->NMR.rwNMR_config.getBitBlockBatchesSize(); id++)
	{
		dataIndex = id + batch_offset;
		if(dataIndex < this->NMR.bitBlock.numberOfBlocks) 
		{
			data[id] = NMR.bitBlock.blocks[id + batch_offset];
		}
		else
		{
			data[id] = 0.0;
		}
	}

	// send data to other processes
	for(uint proc = 0; proc < this->mpi_processes; proc++)
	{
	    if(proc != this->mpi_rank)
	    {
	        // MPI send
	        int destination = proc;
	        int tag = 200 + _batch; 
	        int data_size = this->NMR.rwNMR_config.getBitBlockBatchesSize();
	        MPI_Send(&data,
	                 data_size,
	                 MPI_UINT64_T,
	                 destination,
	                 tag,
	                 MPI_COMM_WORLD);
		}
	}

	// Notify other processes that batch was sent
	(*this).notifyState(this->NMR.rwNMR_config.getBatchTag());

}

void NMR_Network::receiveBitBlockInBatches()
{
	// set blocks array
	NMR.bitBlock.blocks = new uint64_t[this->NMR.bitBlock.numberOfBlocks];

	// set number of batches based on batch size
	uint batches = (uint) ceil(this->NMR.bitBlock.numberOfBlocks / (double) this->NMR.rwNMR_config.getBitBlockBatchesSize());

	for(uint batch = 0; batch < batches; batch++)
	{
		cout << "[" << this->mpi_rank << "]" << " ~ receiving batch " << batch+1 << "/" << batches << endl;
		(*this).receiveBatch(batch);
	}
}

void NMR_Network::receiveBatch(uint _batch)
{
	// Wait for a message with tag 
	int source = 0; // master thread
	MPI_Status status;
	int batch_tag = 200 + _batch;
	MPI_Probe(source, batch_tag, MPI_COMM_WORLD, &status);

	// Get the number of elements in the message
	int data_size;
	MPI_Get_elements(&status, MPI_UINT64_T, &data_size);


	// Receive the message
	uint64_t data[this->NMR.rwNMR_config.getBitBlockBatchesSize()];
	MPI_Recv(&data, 
			 data_size, 
			 MPI_UINT64_T, 
			 status.MPI_SOURCE, 
			 batch_tag,
		     MPI_COMM_WORLD,
		     &status);

	// unwrap 'BitBlock' data		
	// set offset in data access
	uint batch_offset = _batch * this->NMR.rwNMR_config.getBitBlockBatchesSize();
	uint dataIndex;
	for(uint id = 0; id < data_size; id++)
	{
		dataIndex = id + batch_offset;
		if(dataIndex < this->NMR.bitBlock.numberOfBlocks)
		{
			this->NMR.bitBlock.blocks[dataIndex] = data[id];
		}
	}

	// Notify others that batch was received
	(*this).notifyState(this->NMR.rwNMR_config.getBatchTag());
}




void NMR_Network::receiveBitBlock()
{
	uint batches = (uint) ceil(this->NMR.bitBlock.numberOfBlocks / (double) this->NMR.rwNMR_config.getBitBlockBatchesSize());
	cout << "[" << this->mpi_rank << "]" << " ~blocks = " << this->NMR.bitBlock.numberOfBlocks << endl;	
	cout << "[" << this->mpi_rank << "]" << " ~ batches = " << batches << endl;

	// MPI receive          
	int source = 0; // master thread
	int tag = 2; 

	// Wait for a message with tag 
	MPI_Status status;
	MPI_Probe(source, tag, MPI_COMM_WORLD, &status);

	// Get the number of elements in the message
	int data_size;
	MPI_Get_elements(&status, MPI_UINT64_T, &data_size);

	// Allocate buffer of appropriate size
	uint64_t data[this->NMR.bitBlock.numberOfBlocks];

	// Receive the message
	MPI_Recv(&data, 
			 data_size, 
			 MPI_UINT64_T, 
			 status.MPI_SOURCE, 
			 tag,
		     MPI_COMM_WORLD,
		     &status);

	// unwrap 'BitBlock' data
	NMR.bitBlock.blocks = new uint64_t[data_size];
	for(uint id = 0; id < data_size; id++)
	{
		this->NMR.bitBlock.blocks[id] = data[id];
	}
}

void NMR_Network::exchange_NMRT2()
{
	if(mpi_rank == 0)
	{
		send_NMRT2();
	} else
	{
		receive_NMRT2();
	}
}

void NMR_Network::send_NMRT2()
{
	// wrap 'NMR T2' data
	mpi_nmr_T2 NMR_T2;
	for(uint i = 0; i < this->NMR.rwNMR_config.getNMRT2Size(); i++)
	{
		NMR_T2.value[i] = NMR.T2_input[i];
	}

	for(uint proc = 0; proc < this->mpi_processes; proc++)
    {
        if(proc != this->mpi_rank)
        {
            // MPI send
            int destination = proc;
            int tag = 3; 
            int data_size = 1;
            MPI_Send(&NMR_T2,
                     data_size,
                     MPI_NMRT2,
                     destination,
                     tag,
                     MPI_COMM_WORLD);
		}
    }

}

void NMR_Network::receive_NMRT2()
{
	// MPI receive   
	mpi_nmr_T2 NMR_T2;       
	int source = 0; // master thread
	int tag = 3; 
	int data_size = 1;
	MPI_Status status;
	MPI_Recv(&NMR_T2,
		     data_size,
		     MPI_NMRT2,
		     source,
		     tag,
		     MPI_COMM_WORLD,
		     &status);

	// make sure that NMR T2 vector is empty
	if(NMR.T2_input.size() > 0) 
		NMR.T2_input.clear();

	// unwrap 'NMR T2' data
	for(uint id = 0; id < this->NMR.rwNMR_config.getNMRT2Size(); id++)
	{
		NMR.T2_input.push_back(NMR_T2.value[id]);
	}
}

void NMR_Network::notifyState(int _state)
{
	(*this).sendStateNotification(_state);
	(*this).receiveStateNotifications(_state);
}

void NMR_Network::sendStateNotification(int _state)
{
	for(uint proc = 0; proc < this->mpi_processes; proc++)
    {
        if(proc != this->mpi_rank)
        {
            // MPI send
            int msg = 0;
            int destination = proc;
            int tag = _state; 
            int data_size = 1;
            MPI_Send(&msg,
                     data_size,
                     MPI_INT,
                     destination,
                     tag,
                     MPI_COMM_WORLD);
		}
    }
}


void NMR_Network::receiveStateNotifications(int _state)
{
	// receive notification that other processes are ready
	for(uint proc = 0; proc < this->mpi_processes; proc++)
	{
		if(proc != this->mpi_rank)
		{
			// MPI send
			int msg = 0;
			int source = proc;
			int tag = _state; 
			int data_size = 1;
			MPI_Status status;
			MPI_Recv(&msg,
					 data_size,
					 MPI_INT,
					 source,
					 tag,
					 MPI_COMM_WORLD,
					 &status);
		}
	}
}

void NMR_Network::time()
{
	sleep(0.5);
	(*this).printTime();
	sleep(0.5);
}