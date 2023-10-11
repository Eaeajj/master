#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define	N (1024*1024)


int main(int argc, char *argv[]) {
    int		deviceCount;
    cudaDeviceProp	devProp;
    
    cudaGetDeviceCount ( &deviceCount );
    printf             ( "Found %d devices\n", deviceCount );
    
    for ( int device = 0; device < deviceCount; device++ ) {
        cudaGetDeviceProperties ( &devProp, device );		
        printf ( "Device %d\n", device );
        printf ( "Compute capability      : %d.%d\n", devProp.major, devProp.minor );
        printf ( "Name                    : %s\n", devProp.name );
        printf ( "Total Global Memory     : %uB, %lfGB\n", devProp.totalGlobalMem, (double)devProp.totalGlobalMem / 1024 / 1024 / 1024 );
        printf ( "Shared memory per block : %dB, %dKiB\n", devProp.sharedMemPerBlock, devProp.sharedMemPerBlock / 1024 );
        printf ( "Registers per block     : %d\n", devProp.regsPerBlock );
        printf ( "Warp size               : %d\n", devProp.warpSize );
        printf ( "Max threads per block   : %d\n", devProp.maxThreadsPerBlock );
        printf ( "Total constant memory   : %d\n", devProp.totalConstMem );
        printf ( "Max blocks per MP       : %d\n", devProp.maxBlocksPerMultiProcessor );
        printf ( "Max threads per MP      : %d\n", devProp.maxThreadsPerMultiProcessor );
        printf ( "Max threads Dim         : %d\n", devProp.maxThreadsDim );
        printf ( "Count of MPs            : %d\n", devProp.multiProcessorCount );
    }
    return 0;
}