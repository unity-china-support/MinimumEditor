#include "commonCL.h"
#include "directLighting.h"

__kernel void gatherProcessedLightRays(
    // Inputs
    /*00*/ __global const ExpandedTexelInfo* restrict expandedTexelBuffer,
    /*01*/ __global const uint*   restrict            expandedTexelCountBuffer,
    /*02*/ const int                                  radeonRaysExpansionPass,
#ifdef PROBES
    /*03*/ const int                                  numProbes,
    /*04*/ __global const float4* restrict            probeSHExpandedBuffer,
    /*05*/ __global const float4* restrict            probeOcclusionExpandedBuffer,
    /*06*/ __global float4* restrict                  outputProbeDirectSHData,
    /*07*/ __global float4* restrict                  outputProbeOcclusion,
    /*08*/ __global float4* restrict                  outputProbeIndirectSHData
#else
    /*03*/ const int                                  lightmapMode,
    /*04*/ __global const float3* restrict            lightingExpandedBuffer,
    /*05*/ __global const float4* restrict            shadowmaskExpandedBuffer,//when gathering indirect .x will contain AO and .y will contain Validity
    /*06*/ __global const float4* restrict            directionalExpandedBuffer,
    /*07*/ __global float4* restrict                  outputDirectLightingBuffer,
    /*08*/ __global float4* restrict                  outputShadowmaskFromDirectBuffer,
    /*09*/ __global float4* restrict                  outputDirectionalFromDirectBuffer,
    /*10*/ __global float4* restrict                  outputIndirectLightingBuffer,
    /*11*/ __global float4* restrict                  outputEnvironmentLightingBuffer,
    /*12*/ __global float4* restrict                  outputDirectionalFromGiBuffer,
    /*13*/ __global float* restrict                   outputAoBuffer,
    /*14*/ __global float* restrict                   outputValidityBuffer
#endif
    KERNEL_VALIDATOR_BUFFERS_DEF
)
{
    const uint expandedTexelInfoIdx = get_global_id(0);
    const uint numExpandedTexels = INDEX_SAFE(expandedTexelCountBuffer, 0);
    if (expandedTexelInfoIdx < numExpandedTexels)
    {
        const ExpandedTexelInfo expandedTexelInfo = INDEX_SAFE(expandedTexelBuffer, expandedTexelInfoIdx);
        const int numRays = expandedTexelInfo.numRays;
        const int raysOffset = expandedTexelInfo.firstRaysOffset;
        const uint originalTexelIndex = expandedTexelInfo.originalTexelIndex;
#ifdef PROBES
        float4 probeOcclusion = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = 0; i < numRays; ++i)
            probeOcclusion += INDEX_SAFE(probeOcclusionExpandedBuffer, raysOffset + i);

        float4 outSH[SH_COEFF_COUNT];
        for (int coeff = 0; coeff < SH_COEFF_COUNT; ++coeff)
            outSH[coeff] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = 0; i < numRays; ++i)
        {
            int dataPosition = (raysOffset + i) * SH_COEFF_COUNT;
            for (int coeff = 0; coeff < SH_COEFF_COUNT; ++coeff)
            {
                outSH[coeff] += INDEX_SAFE(probeSHExpandedBuffer, dataPosition + coeff);
            }
        }

        // TODO(RadeonRays): memory access is all over the place, make a struct ala SphericalHarmonicsL2 instead of loading/storing with a stride.
        if (radeonRaysExpansionPass == kRRExpansionPass_direct)
        {
            for (int coeff = 0; coeff < SH_COEFF_COUNT; ++coeff)
            {
                INDEX_SAFE(outputProbeDirectSHData, numProbes * coeff + originalTexelIndex) += outSH[coeff];
            }
            INDEX_SAFE(outputProbeOcclusion, originalTexelIndex) += probeOcclusion;
        }
        else
        {
            for (int coeff = 0; coeff < SH_COEFF_COUNT; ++coeff)
            {
                INDEX_SAFE(outputProbeIndirectSHData, numProbes * coeff + originalTexelIndex) += outSH[coeff];
            }
        }
#else
        float4 shadowMask = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float3 lighting = (float3)(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < numRays; ++i)
        {
            // TODO(RadeonRays): only fetch and accumulate shadow mask when needed.
            shadowMask += shadowmaskExpandedBuffer[raysOffset + i];
            lighting += lightingExpandedBuffer[raysOffset + i];
        }

        if (radeonRaysExpansionPass == kRRExpansionPass_direct)
        {
            INDEX_SAFE(outputShadowmaskFromDirectBuffer, originalTexelIndex) += shadowMask;
            INDEX_SAFE(outputDirectLightingBuffer, originalTexelIndex).xyz += lighting;
        }
        else if (radeonRaysExpansionPass == kRRExpansionPass_environment)
        {
            INDEX_SAFE(outputEnvironmentLightingBuffer, originalTexelIndex).xyz += lighting;
        }
        else
        {
            KERNEL_ASSERT(radeonRaysExpansionPass == kRRExpansionPass_indirect);
            INDEX_SAFE(outputIndirectLightingBuffer, originalTexelIndex).xyz += lighting;
            INDEX_SAFE(outputAoBuffer, originalTexelIndex) += shadowMask.x;
            INDEX_SAFE(outputValidityBuffer, originalTexelIndex) += shadowMask.y;
        }

        if (lightmapMode == LIGHTMAPMODE_DIRECTIONAL)
        {
            float4 directionality = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            for (int i = 0; i < numRays; ++i)
            {
                directionality += directionalExpandedBuffer[raysOffset + i];
            }

            if (radeonRaysExpansionPass == kRRExpansionPass_direct)
            {
                INDEX_SAFE(outputDirectionalFromDirectBuffer, originalTexelIndex) += directionality;
            }
            else
            {
                INDEX_SAFE(outputDirectionalFromGiBuffer, originalTexelIndex) += directionality;
            }
        }
#endif
    }
}

__kernel void processLightRays(
    // Inputs
    /*00*/__global const ray*            restrict lightRaysBuffer,
    /*01*/__global const float4*         restrict positionsWSBuffer,
    /*02*/__global const LightBuffer*    restrict directLightsBuffer,
    /*03*/__global const LightSample*    restrict lightSamples,
    /*04*/__global const float4*         restrict lightOcclusionBuffer,
    /*05*/__global const float*          restrict angularFalloffLUT_buffer,
    /*06*/__global const float*          restrict distanceFalloffs_buffer,
    /*07*/__global const uint*           restrict lightRaysCountBuffer,
#ifdef PROBES
    /*08*/const int                               numProbes,
    /*09*/const int                               totalSampleCount,
    /*10*/__global const float4*         restrict inputLightIndices,
    // Outputs
    /*11*/__global float4*               restrict probeSHExpandedBuffer,
    /*12*/__global float4*               restrict probeOcclusionExpandedBuffer
#else
    /*08*/__global const ExpandedRay*    restrict expandedRaysBuffer,
    /*09*/__global const PackedNormalOctQuad* restrict planeNormalsWSBuffer,
    /*10*/const float                             pushOff,
    /*11*/const int                               lightmapMode,
    /*12*/const int                               superSamplingMultiplier,
    /*13*/ __global const unsigned char* restrict gbufferInstanceIdToReceiveShadowsBuffer,
    // Outputs
    /*14*/__global float4* restrict               shadowmaskExpandedBuffer,
    /*15*/__global float4* restrict               directionalExpandedBuffer,
    /*16*/__global float3* restrict               lightingExpandedBuffer
#endif
    KERNEL_VALIDATOR_BUFFERS_DEF
)
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
    LightSample lightSample = INDEX_SAFE(lightSamples, compactedRayIdx);
    LightBuffer light = INDEX_SAFE(directLightsBuffer, lightSample.lightIdx);

#ifndef PROBES
    const ExpandedRay expandedRay = INDEX_SAFE(expandedRaysBuffer, expandedRayIdx);
    const int texelIndex = expandedRay.texelIndex;
    const int ssIdx = GetSuperSampledIndex(expandedRay.texelIndex, expandedRay.currentSampleCount, superSamplingMultiplier);
    const float4 positionAndGbufferInstanceId = INDEX_SAFE(positionsWSBuffer, ssIdx);
    const int gBufferInstanceId = (int)(floor(positionAndGbufferInstanceId.w));
    const float3 P = positionAndGbufferInstanceId.xyz;
    const float3 planeNormal = DecodeNormal(INDEX_SAFE(planeNormalsWSBuffer, ssIdx));
    const float3 position = P + planeNormal * pushOff;
#else
    const int texelIndex = Ray_GetSourceIndex(GET_PTR_SAFE(lightRaysBuffer, compactedRayIdx));
    const int ssIdx = texelIndex;
    const float3 position = INDEX_SAFE(positionsWSBuffer, ssIdx).xyz;
#endif

    bool useShadows = light.castShadow;
#ifndef PROBES
    useShadows &= INDEX_SAFE(gbufferInstanceIdToReceiveShadowsBuffer, gBufferInstanceId);
#endif
    float4 occlusions4 = useShadows ? INDEX_SAFE(lightOcclusionBuffer, compactedRayIdx) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    const bool hit = occlusions4.w < TRANSMISSION_THRESHOLD;
    if (!hit)
    {
#ifdef PROBES
        const float weight = 1.0 / totalSampleCount;
        if (light.directBakeMode >= kDirectBakeMode_Subtractive)
        {
            int lightIdx = light.probeOcclusionLightIndex;
            const float4 lightIndicesFloat = INDEX_SAFE(inputLightIndices, texelIndex);
            int4 lightIndices = (int4)((int)(lightIndicesFloat.x), (int)(lightIndicesFloat.y), (int)(lightIndicesFloat.z), (int)(lightIndicesFloat.w));
            float4 channelSelector = (float4)((lightIndices.x == lightIdx) ? 1.0f : 0.0f, (lightIndices.y == lightIdx) ? 1.0f : 0.0f, (lightIndices.z == lightIdx) ? 1.0f : 0.0f, (lightIndices.w == lightIdx) ? 1.0f : 0.0f);
            INDEX_SAFE(probeOcclusionExpandedBuffer, expandedRayIdx) += channelSelector * weight;
        }
        else if (light.directBakeMode != kDirectBakeMode_None)
        {
            float4 D = (float4)(INDEX_SAFE(lightRaysBuffer, compactedRayIdx).d.x, INDEX_SAFE(lightRaysBuffer, compactedRayIdx).d.y, INDEX_SAFE(lightRaysBuffer, compactedRayIdx).d.z, 0);
            float3 L = ShadeLight(light, INDEX_SAFE(lightRaysBuffer, compactedRayIdx), position, angularFalloffLUT_buffer, distanceFalloffs_buffer KERNEL_VALIDATOR_BUFFERS);
            accumulateSHExpanded(L, D, weight, probeSHExpandedBuffer, expandedRayIdx KERNEL_VALIDATOR_BUFFERS);
        }
#else
        if (light.directBakeMode >= kDirectBakeMode_OcclusionChannel0)
        {
            float4 channelSelector = (float4)(light.directBakeMode == kDirectBakeMode_OcclusionChannel0 ? 1.0f : 0.0f, light.directBakeMode == kDirectBakeMode_OcclusionChannel1 ? 1.0f : 0.0f, light.directBakeMode == kDirectBakeMode_OcclusionChannel2 ? 1.0f : 0.0f, light.directBakeMode == kDirectBakeMode_OcclusionChannel3 ? 1.0f : 0.0f);
            INDEX_SAFE(shadowmaskExpandedBuffer, expandedRayIdx) += occlusions4.w * channelSelector;
        }
        else if (light.directBakeMode != kDirectBakeMode_None)
        {
            const float3 lighting = occlusions4.xyz * ShadeLight(light, INDEX_SAFE(lightRaysBuffer, compactedRayIdx), position, angularFalloffLUT_buffer, distanceFalloffs_buffer KERNEL_VALIDATOR_BUFFERS);
            INDEX_SAFE(lightingExpandedBuffer, expandedRayIdx).xyz += lighting;

            //compute directionality from direct lighting
            if (lightmapMode == LIGHTMAPMODE_DIRECTIONAL)
            {
                float lum = Luminance(lighting);
                float4 directionality;
                directionality.xyz = INDEX_SAFE(lightRaysBuffer, compactedRayIdx).d.xyz * lum;
                directionality.w = lum;
                INDEX_SAFE(directionalExpandedBuffer, expandedRayIdx) += directionality;
            }
        }
#endif
    }
}
