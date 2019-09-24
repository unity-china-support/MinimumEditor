#include "commonCL.h"

__kernel void preparePathRays(
    //*** output ***
    /*00*/ __global ray* restrict                       pathRaysBuffer_0,
    /*01*/ __global uint* restrict                      activePathCountBuffer_0,
    /*02*/ __global uint* restrict                      totalRayCastBuffer,
    /*03*/ __global float4* restrict                    originalRaysBuffer,
    //*** input ***
    /*04*/ __global const float4* restrict              positionsWSBuffer,
    /*05*/ int                                          lightmapSize,
    /*06*/ int                                          bounce,
    /*07*/ __global const uint* restrict                random_buffer,
    /*08*/ __global const uint* restrict                sobol_buffer,
    /*09*/ __global const float* restrict               goldenSample_buffer,
    /*10*/ int                                          numGoldenSample,
    /*11*/ __global const ExpandedRay* restrict         expandedRaysBuffer,
    /*12*/ __global const uint*        restrict         expandedRaysCountBuffer
#ifndef PROBES
    ,
    /*13*/ __global const PackedNormalOctQuad* restrict interpNormalsWSBuffer,
    /*14*/ __global const PackedNormalOctQuad* restrict planeNormalsWSBuffer,
    /*15*/ float                                        pushOff,
    /*16*/ int                                          superSamplingMultiplier
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

    int expandedPathRayIdx = get_global_id(0), local_idx;
    const uint expandedRayCount = INDEX_SAFE(expandedRaysCountBuffer, 0);
    if (expandedPathRayIdx < expandedRayCount)
    {
        const ExpandedRay expandedRay = INDEX_SAFE(expandedRaysBuffer, expandedPathRayIdx);
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

        // Initialize sampler state
        int dimensionOffset = UNITY_SAMPLE_DIM_SURFACE_OFFSET + bounce * UNITY_SAMPLE_DIMS_PER_BOUNCE;
        uint scramble = GetScramble(expandedRay.texelIndex, expandedRay.currentSampleCount, lightmapSize, random_buffer KERNEL_VALIDATOR_BUFFERS);
        float2 sample2D = GetRandomSample2D(expandedRay.currentSampleCount, dimensionOffset, scramble, sobol_buffer);

#ifdef PROBES
        float3 D = Sample_MapToSphere(sample2D);
        const float3 P = position.xyz;
#else
        const float3 interpNormal = DecodeNormal(INDEX_SAFE(interpNormalsWSBuffer, ssIdx));
        //Map to hemisphere directed toward normal
        float3 D = GetRandomDirectionOnHemisphere(sample2D, scramble, interpNormal, numGoldenSample, goldenSample_buffer);
        const float3 planeNormal = DecodeNormal(INDEX_SAFE(planeNormalsWSBuffer, ssIdx));
        const float3 P = position.xyz + planeNormal * pushOff;

        // if plane normal is too different from interpolated normal, the hemisphere orientation will be wrong and the sample could be under the surface.
        float dotVal = dot(D, planeNormal);
        if (dotVal > 0.0f && !isnan(dotVal))
#endif
        {
            const float kMaxt = 1000000.0f;
            Ray_Init(&r, P, D, kMaxt, 0.f, 0xFFFFFFFF);

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
#if DISALLOW_PATH_RAYS_COMPACTION
        numRayToAdd = get_local_size(0);
#endif
        numRayPreparedSharedMem = atomic_add(GET_PTR_SAFE(activePathCountBuffer_0, 0), numRayToAdd);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int compactedPathRayIndex = numRayPreparedSharedMem + local_idx;
#if DISALLOW_PATH_RAYS_COMPACTION
    compactedPathRayIndex = expandedPathRayIdx;
#else
    // Write the ray out to memory
    if (Ray_IsActive_Private(&r))
#endif
    {
        INDEX_SAFE(originalRaysBuffer, expandedPathRayIdx) = (float4)(r.d.x, r.d.y, r.d.z, 0);
        Ray_SetExpandedIndex(&r, expandedPathRayIdx);
        INDEX_SAFE(pathRaysBuffer_0, compactedPathRayIndex) = r;
    }
}

__kernel void preparePathRaysFromBounce(
    //*** input ***
    /*00*/ __global const ray* restrict                 pathRaysBuffer_0,
    /*01*/ __global const Intersection* restrict        pathIntersectionsBuffer,
    /*02*/ __global const uint* restrict                activePathCountBuffer_0,
    /*03*/ __global const PackedNormalOctQuad* restrict pathLastPlaneNormalBuffer,
    /*04*/ __global const unsigned char* restrict       pathLastNormalFacingTheRayBuffer,
    /*05*/ __global const ExpandedRay* restrict         expandedRaysBuffer,
    //randomization
    /*06*/ int                                 lightmapSize,
    /*07*/ int                                 bounce,
    /*08*/ __global const uint* restrict       random_buffer,
    /*09*/ __global const uint* restrict       sobol_buffer,
    /*10*/ __global const float* restrict      goldenSample_buffer,
    /*11*/ int                                 numGoldenSample,
    /*12*/ float                               pushOff,
    //*** output ***
    /*13*/ __global ray* restrict              pathRaysBuffer_1,
    /*14*/ __global uint* restrict             totalRayCastBuffer,
    /*15*/ __global uint* restrict             activePathCountBuffer_1,
    //*** in/output ***
    /*16*/ __global float4* restrict           pathThroughputBuffer
    KERNEL_VALIDATOR_BUFFERS_DEF
)
{
    __local uint numRayPreparedSharedMem;
    if (get_local_id(0) == 0)
        numRayPreparedSharedMem = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    ray r;// Prepare ray in private memory
    Ray_SetInactive(&r);

    uint compactedPathRayIdx = get_global_id(0), local_idx;
    bool shouldPrepareNewRay = compactedPathRayIdx < INDEX_SAFE(activePathCountBuffer_0, 0);

#if DISALLOW_PATH_RAYS_COMPACTION
    if (Ray_IsInactive(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx)))
    {
        shouldPrepareNewRay = false;
    }
#endif

    ExpandedRay expandedRay;
    int expandedPathRayIdx;
    if (shouldPrepareNewRay)
    {
        KERNEL_ASSERT(Ray_IsActive(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx)));

        expandedPathRayIdx = Ray_GetExpandedIndex(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx));
        expandedRay = INDEX_SAFE(expandedRaysBuffer, expandedPathRayIdx);

        // We did not hit anything, no bounce path ray.
        const bool pathRayHitSomething = INDEX_SAFE(pathIntersectionsBuffer, compactedPathRayIdx).shapeid > 0;
        shouldPrepareNewRay &= pathRayHitSomething;

        // We hit an invalid triangle (from the back, no double sided GI), stop the path.
        const unsigned char isNormalFacingTheRay = INDEX_SAFE(pathLastNormalFacingTheRayBuffer, compactedPathRayIdx);
        shouldPrepareNewRay = (shouldPrepareNewRay && isNormalFacingTheRay);
    }

    // Russian roulette step can terminate the path
    int dimensionOffset;
    uint scramble;
    const bool doRussianRoulette = (bounce >= 1) && shouldPrepareNewRay;
    if (doRussianRoulette || shouldPrepareNewRay)
    {
        dimensionOffset = UNITY_SAMPLE_DIM_SURFACE_OFFSET + bounce * UNITY_SAMPLE_DIMS_PER_BOUNCE;
        scramble = GetScramble(expandedRay.texelIndex, expandedRay.currentSampleCount, lightmapSize, random_buffer KERNEL_VALIDATOR_BUFFERS);
    }

    if (doRussianRoulette)
    {
        float4 pathThroughput = INDEX_SAFE(pathThroughputBuffer, expandedPathRayIdx);
        float p = max(max(pathThroughput.x, pathThroughput.y), pathThroughput.z);
        float rand = GetRandomSample1D(expandedRay.currentSampleCount, dimensionOffset++, scramble, sobol_buffer);

        if (p < rand)
            shouldPrepareNewRay = false;
        else
            INDEX_SAFE(pathThroughputBuffer, expandedPathRayIdx).xyz *= (1 / p);
    }

    if (shouldPrepareNewRay)
    {
        const float3 planeNormal = DecodeNormal(INDEX_SAFE(pathLastPlaneNormalBuffer, compactedPathRayIdx));
        const float t = INDEX_SAFE(pathIntersectionsBuffer, compactedPathRayIdx).uvwt.w;
        float3 position = INDEX_SAFE(pathRaysBuffer_0, compactedPathRayIdx).o.xyz + INDEX_SAFE(pathRaysBuffer_0, compactedPathRayIdx).d.xyz * t;

        const float kMaxt = 1000000.0f;

        // Random numbers
        float2 sample2D = GetRandomSample2D(expandedRay.currentSampleCount, dimensionOffset, scramble, sobol_buffer);

        // Map to hemisphere directed toward plane normal
        float3 D = GetRandomDirectionOnHemisphere(sample2D, scramble, planeNormal, numGoldenSample, goldenSample_buffer);

        // TODO(RadeonRays) gboisse: we're generating some NaN directions somehow, fix it!!
        if (!any(isnan(D)))
        {
            const float3 P = position.xyz + planeNormal * pushOff;
            Ray_Init(&r, P, D, kMaxt, 0.f, 0xFFFFFFFF);
            Ray_SetSourceIndex(&r, expandedRay.texelIndex);
            Ray_SetExpandedIndex(&r, expandedPathRayIdx);
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
#if DISALLOW_PATH_RAYS_COMPACTION
        numRayToAdd = get_local_size(0);
#endif
        numRayPreparedSharedMem = atomic_add(GET_PTR_SAFE(activePathCountBuffer_1, 0), numRayToAdd);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int compactedBouncedPathRayIndex = numRayPreparedSharedMem + local_idx;
#if DISALLOW_PATH_RAYS_COMPACTION
    compactedBouncedPathRayIndex = compactedPathRayIdx;
#else
    // Write the ray out to memory
    if (Ray_IsActive_Private(&r))
#endif
    {
        INDEX_SAFE(pathRaysBuffer_1, compactedBouncedPathRayIndex) = r;
    }
}
