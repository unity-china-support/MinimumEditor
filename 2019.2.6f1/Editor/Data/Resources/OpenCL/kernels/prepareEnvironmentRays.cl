#include "environmentLighting.h"


__kernel void prepareDirectEnvironmentRays(
    // *** output *** //
    OUTPUT_BUF( 0, ray                  , lightRaysBuffer           ),
    OUTPUT_BUF( 1, uint                 , lightRaysCountBuffer      ),
    OUTPUT_BUF( 2, uint                 , totalRayCastBuffer        ),
    // *** input *** //
    INPUT_EL(   3, int                  , lightmapSize              ),
    INPUT_EL(   4, int                  , envFlags                  ),
    INPUT_EL(   5, int                  , numEnvironmentSamples     ),
    INPUT_BUF(  6, PackedNormalOctQuad  , envDirectionsBuffer       ),
    INPUT_BUF(  7, float4               , positionsWSBuffer         ),
    INPUT_BUF(  8, uint                 , random_buffer             ),
    INPUT_BUF(  9, uint                 , sobol_buffer              ),
    INPUT_BUF( 10, ExpandedRay          , expandedRaysBuffer        ),
    INPUT_BUF( 11, uint                 , expandedRaysCountBuffer   )
#   ifndef PROBES
    ,
    INPUT_BUF( 12, PackedNormalOctQuad  , interpNormalsWSBuffer     ),
    INPUT_BUF( 13, PackedNormalOctQuad  , planeNormalsWSBuffer      ),
    INPUT_EL(  14, float                , pushOff                   ),
    INPUT_EL(  15, int                  , superSamplingMultiplier   )
#   endif
    KERNEL_VALIDATOR_BUFFERS_DEF)
{
    __local uint numRayPreparedSharedMem;
    if (get_local_id(0) == 0)
        numRayPreparedSharedMem = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Prepare ray in private memory
    ray r;
    Ray_SetInactive(&r);

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

        //Random numbers
        int    dimensionOffset = UNITY_SAMPLE_DIM_SURFACE_OFFSET;
        uint   scramble = GetScramble(expandedRay.texelIndex, expandedRay.currentSampleCount, lightmapSize, random_buffer KERNEL_VALIDATOR_BUFFERS);
        float3 rand;
               rand.z  = GetRandomSample1D(expandedRay.currentSampleCount, dimensionOffset++, scramble, sobol_buffer);
               rand.xy = GetRandomSample2D(expandedRay.currentSampleCount, dimensionOffset  , scramble, sobol_buffer);

#ifdef PROBES
        float3 P = position.xyz;
        float4 D;
        if (UseEnvironmentMIS(envFlags))
            D = GenerateVolumeEnvironmentRayMIS(numEnvironmentSamples, rand, envDirectionsBuffer KERNEL_VALIDATOR_BUFFERS);
        else
            D = GenerateVolumeEnvironmentRay(rand.xy);
#else
        float3 interpNormal = DecodeNormal(INDEX_SAFE(interpNormalsWSBuffer, ssIdx));
        float3 planeNormal  = DecodeNormal(INDEX_SAFE(planeNormalsWSBuffer, ssIdx));

        float4 D;
        if (UseEnvironmentMIS(envFlags))
            D = GenerateSurfaceEnvironmentRayMIS(numEnvironmentSamples, interpNormal, planeNormal, rand, envDirectionsBuffer KERNEL_VALIDATOR_BUFFERS);
        else
            D = GenerateSurfaceEnvironmentRay(interpNormal, planeNormal, rand.xy);

        float3 P = position.xyz + planeNormal * pushOff;
#endif
        if (D.w != 0.0f)
        {
            Ray_Init(&r, P, D.xyz, DEFAULT_RAY_LENGTH, D.w, DEFAULT_RAY_MASK);

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
        Ray_SetExpandedIndex(&r, expandedRayIdx);
        INDEX_SAFE(lightRaysBuffer, compactedIndex) = r;
    }
}

__kernel void prepareIndirectEnvironmentRays(
    //*** output ***
    OUTPUT_BUF( 0, ray                  , lightRaysBuffer                   ),
    OUTPUT_BUF( 1, uint                 , lightRaysCountBuffer              ),
    OUTPUT_BUF( 2, uint                 , totalRayCastBuffer                ),
    OUTPUT_BUF( 3, uint                 , lightRayIndexToPathRayIndexBuffer ),
    //*** input ***
    INPUT_BUF(  4, ray                  , pathRaysBuffer_0                  ),
    INPUT_BUF(  5, uint                 , activePathCountBuffer_0           ),
    INPUT_BUF(  6, Intersection         , pathIntersectionsBuffer           ),
    INPUT_BUF(  7, PackedNormalOctQuad  , pathLastPlaneNormalBuffer         ),
    INPUT_BUF(  8, unsigned char        , pathLastNormalFacingTheRayBuffer  ),
    INPUT_BUF(  9, PackedNormalOctQuad  , pathLastInterpNormalBuffer        ),
    INPUT_BUF( 10, uint                 , random_buffer                     ),
    INPUT_BUF( 11, uint                 , sobol_buffer                      ),
    INPUT_BUF( 12, PackedNormalOctQuad  , envDirectionsBuffer               ),
    INPUT_BUF( 13, ExpandedRay          , expandedRaysBuffer                ),
    INPUT_EL(  14, int                  , envFlags                          ),
    INPUT_EL(  15, int                  , numEnvironmentSamples             ),
    INPUT_EL(  16, int                  , lightmapSize                      ),
    INPUT_EL(  17, int                  , bounce                            ),
    INPUT_EL(  18, float                , pushOff                           )
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
        int dimensionOffset = UNITY_SAMPLE_DIM_SURFACE_OFFSET + bounce * UNITY_SAMPLE_DIMS_PER_BOUNCE;
        int expandedRayIdx = Ray_GetExpandedIndex(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx));
        const ExpandedRay expandedRay = INDEX_SAFE(expandedRaysBuffer, expandedRayIdx);
        uint scramble = GetScramble(expandedRay.texelIndex, expandedRay.currentSampleCount, lightmapSize, random_buffer KERNEL_VALIDATOR_BUFFERS);

        // Random numbers
        float3 rand;
               rand.z  = GetRandomSample1D(expandedRay.currentSampleCount, dimensionOffset++, scramble, sobol_buffer);
               rand.xy = GetRandomSample2D(expandedRay.currentSampleCount, dimensionOffset  , scramble, sobol_buffer);

        float3 planeNormal  = DecodeNormal(INDEX_SAFE(pathLastPlaneNormalBuffer, compactedPathRayIdx));
        float3 interpNormal = DecodeNormal(INDEX_SAFE(pathLastInterpNormalBuffer, compactedPathRayIdx));

        float4 D;
        if (UseEnvironmentMIS(envFlags))
            D = GenerateSurfaceEnvironmentRayMIS(numEnvironmentSamples, interpNormal, planeNormal, rand, envDirectionsBuffer KERNEL_VALIDATOR_BUFFERS);
        else
            D = GenerateSurfaceEnvironmentRay(interpNormal, planeNormal, rand.xy);

        // TODO(RadeonRays) gboisse: we're generating some NaN directions somehow, fix it!!
        if (D.w != 0.0f && !any(isnan(D)))
        {
            float  t  = INDEX_SAFE(pathIntersectionsBuffer, compactedPathRayIdx).uvwt.w;
            float3 P  = INDEX_SAFE(pathRaysBuffer_0, compactedPathRayIdx).o.xyz + INDEX_SAFE(pathRaysBuffer_0, compactedPathRayIdx).d.xyz * t;
                   P += planeNormal * pushOff;
            Ray_Init(&r, P, D.xyz, DEFAULT_RAY_LENGTH, D.w, DEFAULT_RAY_MASK);

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
        INDEX_SAFE(lightRaysBuffer, compactedLightRayIndex) = r;
        INDEX_SAFE(lightRayIndexToPathRayIndexBuffer, compactedLightRayIndex) = compactedPathRayIdx;
    }
}
