#include "commonCL.h"

__constant ConvergenceOutputData g_clearedConvergenceOutputData = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MIN, INT_MIN, INT_MIN};

__kernel void clearConvergenceData(
    __global ConvergenceOutputData*  convergenceOutputDataBuffer
    KERNEL_VALIDATOR_BUFFERS_DEF
)
{
    INDEX_SAFE(convergenceOutputDataBuffer, 0) = g_clearedConvergenceOutputData;
}

__kernel void calculateConvergenceMap(
    //*** input ***
    /*00*/ __global const float4* restrict  positionsWSBuffer,
    /*01*/ __global const unsigned char* restrict cullingMapBuffer,
    /*02*/ __global const int* restrict     directSampleCountBuffer,
    /*03*/ __global const int* restrict     indirectSampleCountBuffer,
    /*04*/ __global const int* restrict     environmentSampleCountBuffer,
    /*05*/ const int                        maxDirectSamplesPerPixel,
    /*06*/ const int                        maxGISamplesPerPixel,
    /*07*/ const int                        maxEnvSamplesPerPixel,
    /*08*/ __global const unsigned char* restrict occupancyBuffer,
    //*** output ***
    /*09*/ __global ConvergenceOutputData*  convergenceOutputDataBuffer //Should be cleared properly before kernel is running
    KERNEL_VALIDATOR_BUFFERS_DEF
)
{
    __local ConvergenceOutputData dataShared;

    int idx = get_global_id(0);

    if (get_local_id(0) == 0)
        dataShared = g_clearedConvergenceOutputData;

    barrier(CLK_LOCAL_MEM_FENCE);

    const int occupiedSamplesWithinTexel = INDEX_SAFE(occupancyBuffer, idx);

    if (occupiedSamplesWithinTexel != 0)
    {
        atomic_inc(&(dataShared.occupiedTexelCount));

        const bool isTexelVisible = !IsCulled(INDEX_SAFE(cullingMapBuffer, idx));
        if (isTexelVisible)
            atomic_inc(&(dataShared.visibleTexelCount));

        const int directSampleCount = INDEX_SAFE(directSampleCountBuffer, idx);
        atomic_min(&(dataShared.minDirectSamples), directSampleCount);
        atomic_max(&(dataShared.maxDirectSamples), directSampleCount);
        atomic_add(&(dataShared.totalDirectSamples), directSampleCount);

        const int giSampleCount = INDEX_SAFE(indirectSampleCountBuffer, idx);
        atomic_min(&(dataShared.minGISamples), giSampleCount);
        atomic_max(&(dataShared.maxGISamples), giSampleCount);
        atomic_add(&(dataShared.totalGISamples), giSampleCount);

        const int envSampleCount = INDEX_SAFE(environmentSampleCountBuffer, idx);
        atomic_min(&(dataShared.minEnvSamples), envSampleCount);
        atomic_max(&(dataShared.maxEnvSamples), envSampleCount);
        atomic_add(&(dataShared.totalEnvSamples), envSampleCount);

        if (IsGIConverged(giSampleCount, maxGISamplesPerPixel))
        {
            atomic_inc(&(dataShared.convergedGITexelCount));

            if (isTexelVisible)
                atomic_inc(&(dataShared.visibleConvergedGITexelCount));
        }

        if (IsDirectConverged(directSampleCount, maxDirectSamplesPerPixel))
        {
            atomic_inc(&(dataShared.convergedDirectTexelCount));

            if (isTexelVisible)
                atomic_inc(&(dataShared.visibleConvergedDirectTexelCount));
        }

        if (IsEnvironmentConverged(envSampleCount, maxEnvSamplesPerPixel))
        {
            atomic_inc(&(dataShared.convergedEnvTexelCount));

            if (isTexelVisible)
                atomic_inc(&(dataShared.visibleConvergedEnvTexelCount));
        }

    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0)
    {
        atomic_add(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).occupiedTexelCount), dataShared.occupiedTexelCount);
        atomic_add(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).visibleTexelCount), dataShared.visibleTexelCount);
        atomic_min(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).minDirectSamples), dataShared.minDirectSamples);
        atomic_max(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).maxDirectSamples), dataShared.maxDirectSamples);
        atomic_add(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).totalDirectSamples), dataShared.totalDirectSamples);
        atomic_min(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).minGISamples), dataShared.minGISamples);
        atomic_max(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).maxGISamples), dataShared.maxGISamples);
        atomic_add(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).totalGISamples), dataShared.totalGISamples);
        atomic_min(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).minEnvSamples), dataShared.minEnvSamples);
        atomic_max(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).maxEnvSamples), dataShared.maxEnvSamples);
        atomic_add(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).totalEnvSamples), dataShared.totalEnvSamples);
        atomic_add(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).convergedGITexelCount), dataShared.convergedGITexelCount);
        atomic_add(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).visibleConvergedGITexelCount), dataShared.visibleConvergedGITexelCount);
        atomic_add(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).convergedDirectTexelCount), dataShared.convergedDirectTexelCount);
        atomic_add(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).visibleConvergedDirectTexelCount), dataShared.visibleConvergedDirectTexelCount);
        atomic_add(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).convergedEnvTexelCount), dataShared.convergedEnvTexelCount);
        atomic_add(&(INDEX_SAFE(convergenceOutputDataBuffer, 0).visibleConvergedEnvTexelCount), dataShared.visibleConvergedEnvTexelCount);
    }
}
