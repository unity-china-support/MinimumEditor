#include "environmentLighting.h"


__kernel void processDirectEnvironment(
    //*** input ***
    INPUT_BUF( 00, ray                , lightRaysBuffer                   ),
    INPUT_BUF( 01, uint               , lightRaysCountBuffer              ),
    INPUT_BUF( 02, float4             , lightOcclusionBuffer              ),
    INPUT_BUF( 03, float4             , env_mipped_cube_texels_buffer     ),
    INPUT_BUF( 04, int                , env_mip_offsets_buffer            ),
    INPUT_EL(  05, Environment        , envData                           ),
#ifdef PROBES
    INPUT_EL(  06, int                , totalSampleCount                  ),
    //*** output ***
    OUTPUT_BUF(07, float4             , probeSHExpandedBuffer             )
#else
    INPUT_BUF( 06, PackedNormalOctQuad, interpNormalsWSBuffer             ),
    INPUT_EL(  07, int                , lightmapMode                      ),
    INPUT_EL(  08, int                , superSamplingMultiplier           ),
    INPUT_BUF( 09, ExpandedRay        , expandedRaysBuffer                ),
    //*** output ***
    OUTPUT_BUF(10, float4             , directionalExpandedBuffer         ),
    OUTPUT_BUF(11, float3             , lightingExpandedBuffer            )

#endif
    KERNEL_VALIDATOR_BUFFERS_DEF)
{
    uint compactedRayIdx = get_global_id(0);
    if (compactedRayIdx >= INDEX_SAFE(lightRaysCountBuffer, 0))
        return;

#if DISALLOW_LIGHT_RAYS_COMPACTION
    if (Ray_IsInactive(GET_PTR_SAFE(lightRaysBuffer, compactedRayIdx)))
        return;
#endif

    KERNEL_ASSERT(Ray_IsActive(GET_PTR_SAFE(lightRaysBuffer, compactedRayIdx)));
    int expandedRayIdx = Ray_GetExpandedIndex(GET_PTR_SAFE(lightRaysBuffer, compactedRayIdx));

    float4 occlusion = INDEX_SAFE(lightOcclusionBuffer, compactedRayIdx);
    bool   occluded  = occlusion.w < TRANSMISSION_THRESHOLD;

    if (!occluded)
    {
        // Environment intersection
        float4 dir          = INDEX_SAFE(lightRaysBuffer, compactedRayIdx).d;
        float3 color        = make_float3(0.0f, 0.0f, 0.0f);
#ifdef PROBES
        if (UseEnvironmentMIS(envData.flags))
            color = ProcessVolumeEnvironmentRayMIS(dir, envData.envDim, envData.numMips, envData.envmapIntegral, env_mipped_cube_texels_buffer, env_mip_offsets_buffer KERNEL_VALIDATOR_BUFFERS);
        else
            color = ProcessVolumeEnvironmentRay(dir, envData.envDim, envData.numMips, env_mipped_cube_texels_buffer, env_mip_offsets_buffer KERNEL_VALIDATOR_BUFFERS);

        // accumulate environment lighting
        color *= occlusion.xyz;
        KERNEL_ASSERT(totalSampleCount > 0);
        float weight = 1.0f / totalSampleCount;
        accumulateSHExpanded(color.xyz, dir, weight, probeSHExpandedBuffer, expandedRayIdx KERNEL_VALIDATOR_BUFFERS);
#else
        const ExpandedRay expandedRay = INDEX_SAFE(expandedRaysBuffer, expandedRayIdx);
        const int ssIdx = GetSuperSampledIndex(expandedRay.texelIndex, expandedRay.currentSampleCount, superSamplingMultiplier);
        float3 interpNormal = DecodeNormal(INDEX_SAFE(interpNormalsWSBuffer, ssIdx));
        if (UseEnvironmentMIS(envData.flags))
            color = ProcessEnvironmentRayMIS(dir, interpNormal, envData.envDim, envData.numMips, envData.envmapIntegral, env_mipped_cube_texels_buffer, env_mip_offsets_buffer KERNEL_VALIDATOR_BUFFERS);
        else
            color = ProcessEnvironmentRay(dir, envData.envDim, envData.numMips, env_mipped_cube_texels_buffer, env_mip_offsets_buffer KERNEL_VALIDATOR_BUFFERS);

        // accumulate environment lighting
        INDEX_SAFE(lightingExpandedBuffer, expandedRayIdx) += occlusion.xyz * color;

        //compute directionality from indirect
        if (lightmapMode == LIGHTMAPMODE_DIRECTIONAL)
        {
            float  luminance   = Luminance(color);
            float3 scaledDir   = dir.xyz * luminance;
            float4 directional = make_float4(scaledDir.x, scaledDir.y, scaledDir.z, luminance);
            INDEX_SAFE(directionalExpandedBuffer, expandedRayIdx) += directional;
        }
#endif
    }
}

__kernel void processIndirectEnvironment(
    INPUT_BUF(  0, ray                , lightRaysBuffer                  ),
    INPUT_BUF(  1, uint               , lightRaysCountBuffer             ),
    INPUT_BUF(  2, ray                , pathRaysBuffer_0                 ),//Only for kernel assert purpose.
    INPUT_BUF(  3, uint               , activePathCountBuffer_0          ),//Only for kernel assert purpose.
    INPUT_BUF(  4, uint               , lightRayIndexToPathRayIndexBuffer),
    INPUT_BUF(  5, float4             , originalRaysBuffer               ),
    INPUT_BUF(  6, float4             , lightOcclusionBuffer             ),
    INPUT_BUF(  7, float4             , pathThroughputBuffer             ),
    INPUT_BUF(  8, PackedNormalOctQuad, pathLastInterpNormalBuffer       ),
    INPUT_BUF(  9, float4             , env_mipped_cube_texels_buffer    ),
    INPUT_BUF( 10, int                , env_mip_offsets_buffer           ),
    INPUT_EL(  11, Environment        , envData                          ),
#ifdef PROBES
    INPUT_EL(  12, int                , totalSampleCount                 ),
    OUTPUT_BUF(13, float4             , probeSHExpandedBuffer            )
#else
    INPUT_EL(  12, int                , lightmapMode                     ),
    OUTPUT_BUF(13, float3             , lightingExpandedBuffer           ),
    OUTPUT_BUF(14, float4             , directionalExpandedBuffer        )
#endif
    KERNEL_VALIDATOR_BUFFERS_DEF
)
{
    uint compactedLightRayIdx = get_global_id(0);
    bool shouldProcessRay = true;

#if DISALLOW_LIGHT_RAYS_COMPACTION
    if (Ray_IsInactive(GET_PTR_SAFE(lightRaysBuffer, compactedLightRayIdx)))
    {
        shouldProcessRay = false;
    }
#endif

    if (shouldProcessRay && compactedLightRayIdx < INDEX_SAFE(lightRaysCountBuffer, 0))
    {
        KERNEL_ASSERT(Ray_IsActive(GET_PTR_SAFE(lightRaysBuffer, compactedLightRayIdx)));
        const int compactedPathRayIdx = INDEX_SAFE(lightRayIndexToPathRayIndexBuffer, compactedLightRayIdx);
        KERNEL_ASSERT(compactedPathRayIdx < INDEX_SAFE(activePathCountBuffer_0, 0));
        KERNEL_ASSERT(Ray_IsActive(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx)));
        const int expandedRayIdx = Ray_GetExpandedIndex(GET_PTR_SAFE(lightRaysBuffer, compactedLightRayIdx));

        float4 occlusion = INDEX_SAFE(lightOcclusionBuffer, compactedLightRayIdx);
        bool   occluded = occlusion.w < TRANSMISSION_THRESHOLD;
        if (!occluded)
        {
            // Environment intersection
            float3 interpNormal = DecodeNormal(INDEX_SAFE(pathLastInterpNormalBuffer, compactedPathRayIdx));
            float4 dir = INDEX_SAFE(lightRaysBuffer, compactedLightRayIdx).d;
            float3 color;

            if (UseEnvironmentMIS(envData.flags))
                color = ProcessEnvironmentRayMIS(dir, interpNormal, envData.envDim, envData.numMips, envData.envmapIntegral, env_mipped_cube_texels_buffer, env_mip_offsets_buffer KERNEL_VALIDATOR_BUFFERS);
            else
                color = ProcessEnvironmentRay(dir, envData.envDim, envData.numMips, env_mipped_cube_texels_buffer, env_mip_offsets_buffer KERNEL_VALIDATOR_BUFFERS);

            color *= occlusion.xyz * INDEX_SAFE(pathThroughputBuffer, expandedRayIdx).xyz;
            float4 originalRayDirection = INDEX_SAFE(originalRaysBuffer, expandedRayIdx);
#ifdef PROBES
            float weight = 1.0f / totalSampleCount;
            accumulateSHExpanded(color.xyz, originalRayDirection, weight, probeSHExpandedBuffer, expandedRayIdx KERNEL_VALIDATOR_BUFFERS);
#else
            //compute directionality from indirect
            if (lightmapMode == LIGHTMAPMODE_DIRECTIONAL)
            {
                float luminance = Luminance(color);
                originalRayDirection.xyz *= luminance;
                originalRayDirection.w = luminance;
                INDEX_SAFE(directionalExpandedBuffer, expandedRayIdx) += originalRayDirection;
            }
            // accumulate environment lighting
            INDEX_SAFE(lightingExpandedBuffer, expandedRayIdx) += occlusion.xyz * color;
#endif
        }
    }
}
