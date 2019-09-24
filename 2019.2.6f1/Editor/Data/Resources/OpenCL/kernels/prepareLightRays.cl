#include "commonCL.h"
#include "directLighting.h"

static int GetCellIndex(float3 position, float3 gridBias, float3 gridScale, int3 gridDims)
{
    const int3 cellPos = clamp(convert_int3(position * gridScale + gridBias), (int3)0, gridDims - 1);
    return cellPos.x + cellPos.y * gridDims.x + cellPos.z * gridDims.x * gridDims.y;
}

__kernel void prepareRayIndices(
    //output
    /*00*/ __global ExpandedRay* restrict       expandedRaysBuffer,
    /*01*/ __global uint* restrict              expandedRaysCountBuffer,
    /*02*/ __global ExpandedTexelInfo* restrict expandedTexelBuffer,
    /*03*/ __global uint* restrict       expandedTexelCountBuffer,
    //input and output
    /*04*/ __global int* restrict        directSampleCountBuffer,
    /*05*/ __global int* restrict        environmentSampleCountBuffer,
    /*06*/ __global int* restrict        indirectSampleCountBuffer,
    //input
    /*07*/ int                           radeonRaysExpansionPass,
    /*08*/ int                           numRaysToShootPerTexel,
    /*09*/ int                           maxSampleCount,
    /*10*/ int                           maxOutputRayCount
#ifndef PROBES
    ,
    /*11*/ __global const unsigned char* restrict cullingMapBuffer,
    /*12*/ __global const unsigned char* restrict occupancyBuffer,
    /*13*/ int                           shouldUseCullingMap
#endif
    KERNEL_VALIDATOR_BUFFERS_DEF
)
{
    // Initialize local memory
    __local int numExpandedTexelsForThreadGroup;
    __local int threadGroupExpandedTexelOffsetInGlobalMemory;
    __local int numRaysForThreadGroup;
    __local int threadGroupRaysOffsetInGlobalMemory;
    if (get_local_id(0) == 0)
    {
        numExpandedTexelsForThreadGroup = 0;
        threadGroupExpandedTexelOffsetInGlobalMemory = 0;
        numRaysForThreadGroup = 0;
        threadGroupRaysOffsetInGlobalMemory = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint idx = get_global_id(0);
    int numRaysToPrepare = numRaysToShootPerTexel;
#if DISALLOW_RAY_EXPANSION
    numRaysToPrepare = 1;
#endif

    // STEP 1 : Determine if the texel is active (i.e. occupied && visible).
#ifndef PROBES
    const int occupiedSamplesWithinTexel = INDEX_SAFE(occupancyBuffer, idx);
    if (occupiedSamplesWithinTexel == 0)
        numRaysToPrepare = 0;
    if (shouldUseCullingMap && numRaysToPrepare && IsCulled(INDEX_SAFE(cullingMapBuffer, idx)))
        numRaysToPrepare = 0;
#endif

    // STEP 2 : Compute how many rays we want to shoot for the active texels.
    int currentSampleCount;
    if (numRaysToPrepare)
    {
        if (radeonRaysExpansionPass == kRRExpansionPass_direct)
        {
            currentSampleCount = INDEX_SAFE(directSampleCountBuffer, idx);
        }
        else if (radeonRaysExpansionPass == kRRExpansionPass_environment)
        {
            currentSampleCount = INDEX_SAFE(environmentSampleCountBuffer, idx);
        }
        else if (radeonRaysExpansionPass == kRRExpansionPass_indirect)
        {
            currentSampleCount = INDEX_SAFE(indirectSampleCountBuffer, idx);
        }

        KERNEL_ASSERT(maxSampleCount >= currentSampleCount);
        int samplesLeftBeforeConvergence = max(maxSampleCount - currentSampleCount, 0);
        numRaysToPrepare = min(samplesLeftBeforeConvergence, numRaysToPrepare);
    }

    // STEP 3 : Compute rays write offsets and init the rays indices.
    int rayOffsetInThreadGroup = 0;
    if (numRaysToPrepare)
        rayOffsetInThreadGroup = atomic_add(&numRaysForThreadGroup, numRaysToPrepare);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0)
    {
#if DISALLOW_RAY_EXPANSION
        numRaysForThreadGroup = get_local_size(0);
#endif
        //Note: ExpandedRaysCountBuffer will be potentially bigger than the size of expandedRaysBuffer. However this is fine
        //as we will only dispatch the following kernel with numthread = ray buffer size.
        threadGroupRaysOffsetInGlobalMemory = atomic_add(GET_PTR_SAFE(expandedRaysCountBuffer, 0), numRaysForThreadGroup);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // STEP 4 : Write the rays texel index out (avoiding writing more rays than the buffer can hold).
    int threadGlobalRayOffset = threadGroupRaysOffsetInGlobalMemory + rayOffsetInThreadGroup;
    int maxNumRaysThisThreadCanPrepare = max(maxOutputRayCount - threadGlobalRayOffset, 0);
    numRaysToPrepare = min(maxNumRaysThisThreadCanPrepare, numRaysToPrepare);
#if DISALLOW_RAY_EXPANSION
    ExpandedRay expandedRay;
    expandedRay.texelIndex = numRaysToPrepare ? idx : -1;//-1 marks a texel we should not cast a ray to (invalid or culled)
    expandedRay.currentSampleCount = currentSampleCount;
    INDEX_SAFE(expandedRaysBuffer, idx) = expandedRay;
#else
    for (int i = 0; i < numRaysToPrepare; ++i)
    {
        ExpandedRay expandedRay;
        expandedRay.texelIndex = idx;
        expandedRay.currentSampleCount = currentSampleCount + i;
        INDEX_SAFE(expandedRaysBuffer, threadGlobalRayOffset + i) = expandedRay;
    }
#endif

    // STEP 5 : Register expanded texel info for the gather step.
    int expandedTexelOffsetInThreadGroup = 0;
    if (numRaysToPrepare)
        expandedTexelOffsetInThreadGroup = atomic_inc(&numExpandedTexelsForThreadGroup);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0)
    {
        KERNEL_ASSERT(numExpandedTexelsForThreadGroup <= get_local_size(0));
        KERNEL_ASSERT(numRaysForThreadGroup <= (numRaysToShootPerTexel * get_local_size(0)));
        KERNEL_ASSERT(numRaysForThreadGroup >= numExpandedTexelsForThreadGroup);
        KERNEL_ASSERT(numExpandedTexelsForThreadGroup <= get_local_size(0));
        threadGroupExpandedTexelOffsetInGlobalMemory = atomic_add(GET_PTR_SAFE(expandedTexelCountBuffer, 0), numExpandedTexelsForThreadGroup);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (numRaysToPrepare)
    {
        ExpandedTexelInfo expandedTexelInfo;
#if DISALLOW_RAY_EXPANSION
        expandedTexelInfo.firstRaysOffset = idx;
#else
        expandedTexelInfo.firstRaysOffset = threadGlobalRayOffset;
#endif
        expandedTexelInfo.numRays = numRaysToPrepare;
        expandedTexelInfo.originalTexelIndex = idx;
        KERNEL_ASSERT(threadGroupExpandedTexelOffsetInGlobalMemory < get_global_size(0));
        KERNEL_ASSERT((threadGroupExpandedTexelOffsetInGlobalMemory + expandedTexelOffsetInThreadGroup)< get_global_size(0));
        KERNEL_ASSERT(expandedTexelOffsetInThreadGroup < get_local_size(0));
        INDEX_SAFE(expandedTexelBuffer, threadGroupExpandedTexelOffsetInGlobalMemory + expandedTexelOffsetInThreadGroup) = expandedTexelInfo;

        //increment sample count
        if (radeonRaysExpansionPass == kRRExpansionPass_direct)
        {
            INDEX_SAFE(directSampleCountBuffer, idx) += numRaysToPrepare;
        }
        else if (radeonRaysExpansionPass == kRRExpansionPass_environment)
        {
            INDEX_SAFE(environmentSampleCountBuffer, idx) += numRaysToPrepare;
        }
        else if (radeonRaysExpansionPass == kRRExpansionPass_indirect)
        {
            INDEX_SAFE(indirectSampleCountBuffer, idx) += numRaysToPrepare;
        }
    }
}

//Preparing shadowRays for direct lighting.
__kernel void prepareLightRays(
    //outputs
    /*00*/ __global ray*               restrict lightRaysBuffer,
    /*01*/ __global LightSample*       restrict lightSamples,
    /*02*/ __global uint*              restrict totalRayCastBuffer,
    /*03*/ __global uint*              restrict lightRaysCountBuffer,
    //inputs
    /*04*/ __global const float4*      restrict positionsWSBuffer,
    /*05*/ __global const LightBuffer* restrict directLightsBuffer,
    /*06*/ __global const int*         restrict directLightsOffsetBuffer,
    /*07*/ __global const int*         restrict directLightsCountPerCellBuffer,
    /*08*/ const float3                         lightGridBias,
    /*09*/ const float3                         lightGridScale,
    /*10*/ const int3                           lightGridDims,
    /*11*/ const int                            lightmapSize,
    /*12*/ __global const uint*        restrict random_buffer,
    /*13*/ __global const uint*        restrict sobol_buffer,
    /*14*/ __global const uint*        restrict expandedRaysCountBuffer,
    /*15*/ const int                            lightIndexInCell,
    /*16*/ __global const ExpandedRay* restrict expandedRaysBuffer
#ifndef PROBES
    ,
    /*16*/ __global const PackedNormalOctQuad* restrict interpNormalsWSBuffer,
    /*17*/ const float                          pushOff,
    /*18*/ const int                            superSamplingMultiplier
#endif
    KERNEL_VALIDATOR_BUFFERS_DEF
)
{
    __local uint numRayPreparedSharedMem;
    if (get_local_id(0) == 0)
        numRayPreparedSharedMem = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Prepare ray in private memory
    ray r;
    Ray_SetInactive(&r);
    LightSample lightSample;

    int expandedRayIdx = get_global_id(0), local_idx;
    const uint expandedRayCount = INDEX_SAFE(expandedRaysCountBuffer, 0);
    if (expandedRayIdx < expandedRayCount)
    {
        const ExpandedRay expandedRay = INDEX_SAFE(expandedRaysBuffer, expandedRayIdx);
#if DISALLOW_RAY_EXPANSION
        if (expandedRay.texelIndex >= 0)
        {
#endif
#ifndef PROBES
        int ssIdx = GetSuperSampledIndex(expandedRay.texelIndex, expandedRay.currentSampleCount, superSamplingMultiplier);
#else
        int ssIdx = expandedRay.texelIndex;
#endif
        float4 position = INDEX_SAFE(positionsWSBuffer, ssIdx);
        AssertPositionIsOccupied(position KERNEL_VALIDATOR_BUFFERS);

        const int cellIdx = GetCellIndex(position.xyz, lightGridBias, lightGridScale, lightGridDims);
        const int lightCountInCell = INDEX_SAFE(directLightsCountPerCellBuffer, cellIdx);

        // If we already did all the lights in the cell bail out
        if (lightIndexInCell < lightCountInCell)
        {
            // Select a light in a round robin fashion (no need for pdf)
            lightSample.lightIdx = INDEX_SAFE(directLightsOffsetBuffer, cellIdx) + lightIndexInCell;
            lightSample.lightPdf = 1.0f;
            const LightBuffer light = INDEX_SAFE(directLightsBuffer, lightSample.lightIdx);

            // Initialize sampler state
            uint scramble = GetScramble(expandedRay.texelIndex, expandedRay.currentSampleCount, lightmapSize, random_buffer KERNEL_VALIDATOR_BUFFERS);
            float2 sample2D = GetRandomSample2D(expandedRay.currentSampleCount, UNITY_SAMPLE_DIM_CAMERA_OFFSET + lightIndexInCell, scramble, sobol_buffer);

            // Generate the shadow ray. This might be an inactive ray (in case of back facing surfaces or out of cone angle for spots).
#ifdef PROBES
            float3 notUsed3 = (float3)(0, 0, 0);
            PrepareShadowRay(light, sample2D, position.xyz, notUsed3, 0, false, &r);
#else
            float3 normal = DecodeNormal(INDEX_SAFE(interpNormalsWSBuffer, ssIdx));
            PrepareShadowRay(light, sample2D, position.xyz, normal, pushOff, false, &r);
#endif
            // Set the index so we can map to the originating texel/probe
            Ray_SetSourceIndex(&r, expandedRay.texelIndex);
        }
#if DISALLOW_RAY_EXPANSION
        }
#endif
    }

    // Threads synchronization for compaction
    if (Ray_IsActive_Private(&r))
    {
        local_idx = atomic_inc(&numRayPreparedSharedMem);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0)
    {
        atomic_add(GET_PTR_SAFE(totalRayCastBuffer, 0), numRayPreparedSharedMem);
        int numRayToAdd = numRayPreparedSharedMem;
#if DISALLOW_LIGHT_RAYS_COMPACTION
        numRayToAdd = get_local_size(0);
#endif
        numRayPreparedSharedMem = atomic_add(GET_PTR_SAFE(lightRaysCountBuffer, 0), numRayToAdd);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int compactedIndex = numRayPreparedSharedMem + local_idx;
#if DISALLOW_LIGHT_RAYS_COMPACTION
    compactedIndex = expandedRayIdx;
#else
    // Write the ray out to memory
    if (Ray_IsActive_Private(&r))
#endif
    {
        INDEX_SAFE(lightSamples, compactedIndex) = lightSample;
        Ray_SetExpandedIndex(&r, expandedRayIdx);
        INDEX_SAFE(lightRaysBuffer, compactedIndex) = r;
    }
}

//Preparing shadowRays for indirect lighting.
__kernel void prepareLightRaysFromBounce(
    //*** input ***
    /*00*/ __global const LightBuffer*         restrict indirectLightsBuffer,
    /*01*/ __global const int*                 restrict indirectLightsOffsetBuffer,
    /*02*/ __global const int*                 restrict indirectLightsDistribution,
    /*03*/ __global const int*                 restrict indirectLightDistributionOffsetBuffer,
    /*04*/ __global const bool*                restrict usePowerSamplingBuffer,
    /*05*/ const float3                        lightGridBias,
    /*06*/ const float3                        lightGridScale,
    /*07*/ const int3                          lightGridDims,
    /*08*/ __global const ray*                 restrict pathRaysBuffer_0,
    /*09*/ __global const Intersection*        restrict pathIntersectionsBuffer,
    /*10*/ __global const PackedNormalOctQuad* restrict pathLastInterpNormalBuffer,
    /*11*/ __global const unsigned char*       restrict pathLastNormalFacingTheRayBuffer,
    /*12*/ const int                           lightmapSize,
    /*13*/ const int                           bounce,
    /*14*/ __global const uint*                restrict random_buffer,
    /*15*/ __global const uint*                restrict sobol_buffer,
    /*16*/ const float                                  pushOff,
    /*17*/ __global const uint*                restrict activePathCountBuffer_0,
    /*18*/ __global const ExpandedRay*         restrict expandedRaysBuffer,
    //*** output ***
    /*19*/ __global ray*                       restrict lightRaysBuffer,
    /*20*/ __global LightSample*               restrict lightSamples,
    /*21*/ __global uint*                      restrict lightRayIndexToPathRayIndexBuffer,
    /*22*/ __global uint*                      restrict totalRayCastBuffer,
    /*23*/ __global uint*                      restrict lightRaysCountBuffer
    KERNEL_VALIDATOR_BUFFERS_DEF
)
{
    // Initialize local memory
    __local uint numRayPreparedSharedMem;
    if (get_local_id(0) == 0)
        numRayPreparedSharedMem = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Prepare ray in private memory
    ray r;
    Ray_SetInactive(&r);
    LightSample lightSample;

    // Should we prepare a light ray?
    int compactedPathRayIdx = get_global_id(0), local_idx;
    bool shouldPrepareNewRay = compactedPathRayIdx < INDEX_SAFE(activePathCountBuffer_0, 0);

#if DISALLOW_PATH_RAYS_COMPACTION
    if (Ray_IsInactive(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx)))
    {
        shouldPrepareNewRay = false;
    }
#endif

    if (shouldPrepareNewRay)
    {
        KERNEL_ASSERT(Ray_IsActive(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx)));

        // We did not hit anything, no light ray.
        const bool pathRayHitSomething = INDEX_SAFE(pathIntersectionsBuffer, compactedPathRayIdx).shapeid > 0;
        shouldPrepareNewRay &= pathRayHitSomething;

        // We hit an invalid triangle (from the back, no double sided GI), no light ray.
        const bool isNormalFacingTheRay = INDEX_SAFE(pathLastNormalFacingTheRayBuffer, compactedPathRayIdx);
        shouldPrepareNewRay &= isNormalFacingTheRay;
    }

    // Prepare the shadow ray
    if (shouldPrepareNewRay)
    {
        const float3 surfaceNormal = DecodeNormal(INDEX_SAFE(pathLastInterpNormalBuffer, compactedPathRayIdx));
        const float t = INDEX_SAFE(pathIntersectionsBuffer, compactedPathRayIdx).uvwt.w;
        const float3 surfacePosition = INDEX_SAFE(pathRaysBuffer_0, compactedPathRayIdx).o.xyz + t * INDEX_SAFE(pathRaysBuffer_0, compactedPathRayIdx).d.xyz;

        // Retrieve the light distribution at the shading site
        const int cellIdx = GetCellIndex(surfacePosition, lightGridBias, lightGridScale, lightGridDims);
        __global const int *lightDistributionPtr = GET_PTR_SAFE(indirectLightsDistribution, INDEX_SAFE(indirectLightDistributionOffsetBuffer, cellIdx));
        const int lightDistribution = *lightDistributionPtr; // safe to dereference, as GET_PTR_SAFE above does the validation

        // If there is no light in the cell, bail out
        if (lightDistribution)
        {
            int expandedRayIdx = Ray_GetExpandedIndex(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx));
            const ExpandedRay expandedRay = INDEX_SAFE(expandedRaysBuffer, expandedRayIdx);

            // Initialize sampler state
            uint dimension = UNITY_SAMPLE_DIM_CAMERA_OFFSET + bounce * UNITY_SAMPLE_DIMS_PER_BOUNCE + UNITY_SAMPLE_DIM_SURFACE_OFFSET;
            uint scramble = GetScramble(expandedRay.texelIndex, expandedRay.currentSampleCount, lightmapSize, random_buffer KERNEL_VALIDATOR_BUFFERS);

            // Select a light
            float sample1D = GetRandomSample1D(expandedRay.currentSampleCount, dimension++, scramble, sobol_buffer);
            if (INDEX_SAFE(usePowerSamplingBuffer, UsePowerSamplingBufferSlot_PowerSampleEnabled))
            {
                float selectionPdf;
                lightSample.lightIdx = INDEX_SAFE(indirectLightsOffsetBuffer, cellIdx) + Distribution1D_SampleDiscrete(sample1D, lightDistributionPtr, &selectionPdf);
                lightSample.lightPdf = selectionPdf;
            }
            else
            {
                const int offset = min(lightDistribution - 1, (int)(sample1D * (float)lightDistribution));
                lightSample.lightIdx = INDEX_SAFE(indirectLightsOffsetBuffer, cellIdx) + offset;
                lightSample.lightPdf = 1.0f / lightDistribution;
            }

            // Generate the shadow ray
            const LightBuffer light = INDEX_SAFE(indirectLightsBuffer, lightSample.lightIdx);
            float2 sample2D = GetRandomSample2D(expandedRay.currentSampleCount, dimension++, scramble, sobol_buffer);
            PrepareShadowRay(light, sample2D, surfacePosition, surfaceNormal, pushOff, false, &r);

            // Set the index so we can map to the originating texel/probe
            Ray_SetSourceIndex(&r, expandedRay.texelIndex);
            Ray_SetExpandedIndex(&r, expandedRayIdx);
        }
    }

    // Threads synchronization for compaction
    if (Ray_IsActive_Private(&r))
    {
        local_idx = atomic_inc(&numRayPreparedSharedMem);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0)
    {
        atomic_add(GET_PTR_SAFE(totalRayCastBuffer, 0), numRayPreparedSharedMem);
        int numRayToAdd = numRayPreparedSharedMem;
#if DISALLOW_LIGHT_RAYS_COMPACTION
        numRayToAdd = get_local_size(0);
#endif
        numRayPreparedSharedMem = atomic_add(GET_PTR_SAFE(lightRaysCountBuffer, 0), numRayToAdd);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int compactedLightRayIndex = numRayPreparedSharedMem + local_idx;
#if DISALLOW_LIGHT_RAYS_COMPACTION
    compactedLightRayIndex = compactedPathRayIdx;
#else
    // Write the ray out to memory
    if (Ray_IsActive_Private(&r))
#endif
    {
        INDEX_SAFE(lightSamples, compactedLightRayIndex) = lightSample;
        INDEX_SAFE(lightRaysBuffer, compactedLightRayIndex) = r;
        INDEX_SAFE(lightRayIndexToPathRayIndexBuffer, compactedLightRayIndex) = compactedPathRayIdx;
    }
}
