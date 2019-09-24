#include "commonCL.h"
#include "colorSpace.h"
#include "directLighting.h"
#include "emissiveLighting.h"

__constant sampler_t linear2DSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

static void AccumulateLightFromBounce(float3 albedo, float3 directLightingAtHit, int expandedRayIdx, __global float3* lightingExpandedBuffer, int lightmapMode,
    __global float4* directionalExpandedBuffer, float3 direction KERNEL_VALIDATOR_BUFFERS_DEF)
{
    //Purely diffuse surface reflect the unabsorbed light evenly on the hemisphere.
    float3 energyFromHit = albedo * directLightingAtHit;
    INDEX_SAFE(lightingExpandedBuffer, expandedRayIdx).xyz += energyFromHit;

    //compute directionality from indirect
    if (lightmapMode == LIGHTMAPMODE_DIRECTIONAL)
    {
        float lum = Luminance(energyFromHit);

        INDEX_SAFE(directionalExpandedBuffer, expandedRayIdx).xyz += direction * lum;
        INDEX_SAFE(directionalExpandedBuffer, expandedRayIdx).w += lum;
    }
}

__kernel void processLightRaysFromBounce(
    //*** input ***
    //lighting
    /*00*/ __global LightBuffer*    const   indirectLightsBuffer,
    /*01*/ __global LightSample*    const   lightSamples,
    /*02*/__global uint*                    usePowerSamplingBuffer,
    /*03*/ __global float*          const   angularFalloffLUT_buffer,
    /*04*/ __global float* restrict const   distanceFalloffs_buffer,
    //ray
    /*05*/ __global ray*            const   pathRaysBuffer_0,
    /*06*/ __global Intersection*   const   pathIntersectionsBuffer,
    /*07*/ __global ray*            const   lightRaysBuffer,
    /*08*/ __global float4*         const   lightOcclusionBuffer,
    /*09*/ __global float4*         const   pathThroughputBuffer,
    /*10*/ __global uint*           const   lightRayIndexToPathRayIndexBuffer,
    /*11*/ __global uint*           const   lightRaysCountBuffer,
    /*12*/ __global float4*         const   originalRaysBuffer,
#ifdef PROBES
    /*13*/ int                              totalSampleCount,
    //*** output ***
    /*14*/ __global float4*                 probeSHExpandedBuffer
#else
    //directional lightmap
    /*13*/          int                     lightmapMode,
    //*** output ***
    /*14*/ __global float3*                 lightingExpandedBuffer,
    /*15*/ __global float4*                 directionalExpandedBuffer
#endif
    KERNEL_VALIDATOR_BUFFERS_DEF
    )
{
    uint compactedLightRayIdx = get_global_id(0);
    __local int numLightHitCountSharedMem;
    __local int numLightRayCountSharedMem;
    if (get_local_id(0) == 0)
    {
        numLightHitCountSharedMem = 0;
        numLightRayCountSharedMem = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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
        KERNEL_ASSERT(Ray_IsActive(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx)));
        const bool pathRayHitSomething = INDEX_SAFE(pathIntersectionsBuffer, compactedPathRayIdx).shapeid > 0;
        KERNEL_ASSERT(pathRayHitSomething);

        const int texelOrProbeIdx = Ray_GetSourceIndex(GET_PTR_SAFE(lightRaysBuffer, compactedLightRayIdx));
        const int expandedRayIdx = Ray_GetExpandedIndex(GET_PTR_SAFE(lightRaysBuffer, compactedLightRayIdx));
        LightSample lightSample = INDEX_SAFE(lightSamples, compactedLightRayIdx);
        LightBuffer light = INDEX_SAFE(indirectLightsBuffer, lightSample.lightIdx);

        bool useShadows = light.castShadow;
        const float4 occlusions4 = useShadows ? INDEX_SAFE(lightOcclusionBuffer, compactedLightRayIdx) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        const bool  isLightOccludedFromBounce = occlusions4.w < TRANSMISSION_THRESHOLD;

        if (!isLightOccludedFromBounce)
        {
            const float t = INDEX_SAFE(pathIntersectionsBuffer, compactedPathRayIdx).uvwt.w;
            //We need to compute direct lighting on the fly
            float3 surfacePosition = INDEX_SAFE(pathRaysBuffer_0, compactedPathRayIdx).o.xyz + INDEX_SAFE(pathRaysBuffer_0, compactedPathRayIdx).d.xyz * t;
            float3 albedoAttenuation = INDEX_SAFE(pathThroughputBuffer, expandedRayIdx).xyz;

            float3 directLightingAtHit = occlusions4.xyz * ShadeLight(light, INDEX_SAFE(lightRaysBuffer, compactedLightRayIdx), surfacePosition, angularFalloffLUT_buffer, distanceFalloffs_buffer KERNEL_VALIDATOR_BUFFERS) / lightSample.lightPdf;

            // The original direction from which the rays was shot from the probe position
            float4 originalRayDirection = INDEX_SAFE(originalRaysBuffer, expandedRayIdx);
#ifdef PROBES
            float3 L = albedoAttenuation * directLightingAtHit;
            float weight = 4.0 / totalSampleCount;
            accumulateSHExpanded(L, originalRayDirection, weight, probeSHExpandedBuffer, expandedRayIdx KERNEL_VALIDATOR_BUFFERS);
#else
            AccumulateLightFromBounce(albedoAttenuation, directLightingAtHit, expandedRayIdx, lightingExpandedBuffer, lightmapMode, directionalExpandedBuffer, originalRayDirection.xyz KERNEL_VALIDATOR_BUFFERS);
#endif
            atomic_inc(&numLightHitCountSharedMem);
        }
        atomic_inc(&numLightRayCountSharedMem);
    }

    // Collect stats to disable power sampling in pathological case.
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0)
    {
        atomic_add(GET_PTR_SAFE(usePowerSamplingBuffer, UsePowerSamplingBufferSlot_LightHitCount), numLightHitCountSharedMem);
        atomic_add(GET_PTR_SAFE(usePowerSamplingBuffer, UsePowerSamplingBufferSlot_LightRayCount), numLightRayCountSharedMem);
    }
}

__kernel void updatePowerSamplingBuffer(
    /*00*/ int                              resetPowerSamplingBuffer,
    /*01*/__global uint*                    usePowerSamplingBuffer
    KERNEL_VALIDATOR_BUFFERS_DEF
)
{
    if (resetPowerSamplingBuffer)
    {
        //Reset counter and re-enable power sampling
        INDEX_SAFE(usePowerSamplingBuffer, UsePowerSamplingBufferSlot_LightHitCount) = 0;
        INDEX_SAFE(usePowerSamplingBuffer, UsePowerSamplingBufferSlot_LightRayCount) = 0;
        INDEX_SAFE(usePowerSamplingBuffer, UsePowerSamplingBufferSlot_PowerSampleEnabled) = 0xFFFFFFFF;
        return;
    }

    const float kPowerSamplingMinimumRatio = 0.2f;
    const int   kPowerSamplingMinimumRaysCountBeforeDisabling = 100;

    uint totalLightHitCount = INDEX_SAFE(usePowerSamplingBuffer, UsePowerSamplingBufferSlot_LightHitCount);
    uint totalLightRayCount = INDEX_SAFE(usePowerSamplingBuffer, UsePowerSamplingBufferSlot_LightRayCount);
    if (totalLightRayCount > kPowerSamplingMinimumRaysCountBeforeDisabling)
    {
        float ratio = (float)totalLightHitCount / (float)totalLightRayCount;
        if (ratio < kPowerSamplingMinimumRatio)
        {
            //Disable power sampling
            INDEX_SAFE(usePowerSamplingBuffer, UsePowerSamplingBufferSlot_PowerSampleEnabled) = 0;
        }
    }
}

__kernel void processEmissiveAndAOFromBounce(
    //input
    /*00*/ __global ray*            const   pathRaysBuffer_0,
    /*01*/ __global Intersection*   const   pathIntersectionsBuffer,
    /*02*/ __global MaterialTextureProperties* const instanceIdToEmissiveTextureProperties,
    /*03*/ __global float2*         const   geometryUV1sBuffer,
    /*04*/ __global float4*         const   dynarg_texture_buffer,
    /*05*/ __global MeshDataOffsets* const  instanceIdToMeshDataOffsets,
    /*06*/ __global uint*           const   geometryIndicesBuffer,
    /*07*/ __global float4*         const   pathThroughputBuffer,
    /*08*/ __global uint*           const   activePathCountBuffer_0,
    /*09*/ __global const unsigned char* restrict pathLastNormalFacingTheRayBuffer,
    /*10*/ __global float4*         const   originalRaysBuffer,
#ifdef PROBES
    /*11*/ int                              totalSampleCount,
    //output
    /*12*/ __global float4*                 probeSHExpandedBuffer
#else
    /*11*/          int                     lightmapMode,
    /*12*/          float                   aoMaxDistance,
    /*13*/          int                     bounce,
    //output
    /*14*/ __global float3*                 lightingExpandedBuffer,
    /*15*/ __global float4*                 directionalExpandedBuffer,
    /*16*/ __global float4*                 shadowmaskExpandedBuffer //when gathering indirect .x will contain AO and .y will contain Validity
#endif
    KERNEL_VALIDATOR_BUFFERS_DEF
)
{
    uint compactedPathRayIdx = get_global_id(0);

    if (compactedPathRayIdx >= INDEX_SAFE(activePathCountBuffer_0, 0))
        return;

#if DISALLOW_PATH_RAYS_COMPACTION
    if (Ray_IsInactive(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx)))
        return;
#endif

    KERNEL_ASSERT(Ray_IsActive(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx)));
    const int texelOrProbeIdx = Ray_GetSourceIndex(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx));
    const int expandedPathRayIdx = Ray_GetExpandedIndex(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx));
    const bool  hit = INDEX_SAFE(pathIntersectionsBuffer, compactedPathRayIdx).shapeid > 0;

#ifndef PROBES
    const bool shouldAddOneToAOCount = (bounce == 0 && (!hit || INDEX_SAFE(pathIntersectionsBuffer, compactedPathRayIdx).uvwt.w > aoMaxDistance));
    if (shouldAddOneToAOCount)
    {
        INDEX_SAFE(shadowmaskExpandedBuffer, expandedPathRayIdx).x += 1.0f;
    }
#endif

    if (hit)
    {
        AtlasInfo emissiveContribution = FetchEmissionFromRayIntersection(compactedPathRayIdx,
            pathIntersectionsBuffer,
            instanceIdToEmissiveTextureProperties,
            instanceIdToMeshDataOffsets,
            geometryUV1sBuffer,
            geometryIndicesBuffer,
            dynarg_texture_buffer
            KERNEL_VALIDATOR_BUFFERS
        );

        // If hit an invalid triangle (from the back, no double sided GI) we do not apply emissive.
        const unsigned char isNormalFacingTheRay = INDEX_SAFE(pathLastNormalFacingTheRayBuffer, compactedPathRayIdx);

        // The original direction from which the rays was shot
        float4 originalRayDirection = INDEX_SAFE(originalRaysBuffer, expandedPathRayIdx);

#ifdef PROBES
        float3 L = emissiveContribution.color.xyz * INDEX_SAFE(pathThroughputBuffer, expandedPathRayIdx).xyz;
        float weight = 4.0 / totalSampleCount;
        accumulateSHExpanded(L, originalRayDirection, weight, probeSHExpandedBuffer, expandedPathRayIdx KERNEL_VALIDATOR_BUFFERS);
#else
        float3 output = isNormalFacingTheRay * emissiveContribution.color.xyz * INDEX_SAFE(pathThroughputBuffer, expandedPathRayIdx).xyz;

        // Compute directionality from indirect
        if (lightmapMode == LIGHTMAPMODE_DIRECTIONAL)
        {
            float lum = Luminance(output);
            float4 directionality;
            directionality.xyz = originalRayDirection.xyz * lum;
            directionality.w = lum;
            INDEX_SAFE(directionalExpandedBuffer, expandedPathRayIdx) += directionality;
        }


        // Write Result
        INDEX_SAFE(lightingExpandedBuffer, expandedPathRayIdx).xyz += output.xyz;
#endif
    }
}

__kernel void advanceInPathAndAdjustPathProperties(
    //input
    /*00*/ __global ray*            const   pathRaysBuffer_0,
    /*01*/ __global Intersection*   const   pathIntersectionsBuffer,
    /*02*/ __global MaterialTextureProperties* const instanceIdToAlbedoTextureProperties,
    /*03*/ __global MeshDataOffsets* const  instanceIdToMeshDataOffsets,
    /*04*/ __global float2*         const   geometryUV1sBuffer,
    /*05*/ __global uint*           const   geometryIndicesBuffer,
    /*06*/ __global uchar4*         const   albedoTextures_buffer,
    /*07*/ __global uint*           const   activePathCountBuffer_0,
    //in/output
    /*08*/ __global float4*         const   pathThroughputBuffer
    KERNEL_VALIDATOR_BUFFERS_DEF
)
{
    uint compactedPathRayIdx = get_global_id(0);

    if (compactedPathRayIdx >= INDEX_SAFE(activePathCountBuffer_0, 0))
        return;

#if DISALLOW_PATH_RAYS_COMPACTION
    if (Ray_IsInactive(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx)))
        return;
#endif

    KERNEL_ASSERT(Ray_IsActive(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx)));
    const int expandedPathRayIdx = Ray_GetExpandedIndex(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx));
    const bool  hit = INDEX_SAFE(pathIntersectionsBuffer, compactedPathRayIdx).shapeid > 0;
    if (!hit)
        return;

    AtlasInfo albedoAtHit = FetchAlbedoFromRayIntersection(compactedPathRayIdx,
        pathIntersectionsBuffer,
        instanceIdToAlbedoTextureProperties,
        instanceIdToMeshDataOffsets,
        geometryUV1sBuffer,
        geometryIndicesBuffer,
        albedoTextures_buffer
        KERNEL_VALIDATOR_BUFFERS);

    const float throughputAttenuation = dot(albedoAtHit.color.xyz, kAverageFactors);
    INDEX_SAFE(pathThroughputBuffer, expandedPathRayIdx) *= (float4)(albedoAtHit.color.x, albedoAtHit.color.y, albedoAtHit.color.z, throughputAttenuation);
}

__kernel void getNormalsFromLastBounceAndDoValidity(
    //input
    /*00*/ __global const ray* restrict              pathRaysBuffer_0,              // rays from last to current hit
    /*01*/ __global const Intersection* restrict     pathIntersectionsBuffer,       // intersections from last to current hit
    /*02*/ __global const MeshDataOffsets* restrict  instanceIdToMeshDataOffsets,
    /*03*/ __global const Matrix4x4* restrict        instanceIdToInvTransposedMatrices,
    /*04*/ __global const Vector3f_storage* restrict geometryPositionsBuffer,
    /*05*/ __global const PackedNormalOctQuad* restrict geometryNormalsBuffer,
    /*06*/ __global const uint* restrict             geometryIndicesBuffer,
    /*07*/ __global const uint* restrict             activePathCountBuffer_0,
    /*08*/ __global const MaterialTextureProperties* restrict instanceIdToTransmissionTextureProperties,
    /*09*/                int                        validitybufferMode,
    //output
    /*10*/ __global PackedNormalOctQuad* restrict    pathLastPlaneNormalBuffer,
    /*11*/ __global PackedNormalOctQuad* restrict    pathLastInterpNormalBuffer,
    /*12*/ __global unsigned char* restrict          pathLastNormalFacingTheRayBuffer,
    /*13*/ __global float4*  restrict                shadowmaskExpandedBuffer//Used to store validity in .y
    KERNEL_VALIDATOR_BUFFERS_DEF
)
{
    uint compactedPathRayIdx = get_global_id(0);

    if (compactedPathRayIdx >= INDEX_SAFE(activePathCountBuffer_0, 0))
        return;

#if DISALLOW_PATH_RAYS_COMPACTION
    if (Ray_IsInactive(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx)))
        return;
#endif

    KERNEL_ASSERT(Ray_IsActive(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx)));
    if (INDEX_SAFE(pathIntersectionsBuffer, compactedPathRayIdx).shapeid <= 0)
    {
        PackedNormalOctQuad zero;
        zero.x = 0;  zero.y = 0; zero.z = 0; // Will yield a decoded value of float3(0, 0, -1)
        INDEX_SAFE(pathLastPlaneNormalBuffer, compactedPathRayIdx) = zero;
        INDEX_SAFE(pathLastInterpNormalBuffer, compactedPathRayIdx) = zero;
        INDEX_SAFE(pathLastNormalFacingTheRayBuffer, compactedPathRayIdx) = 0;
        return;
    }

    const int instanceId = GetInstanceIdFromIntersection(GET_PTR_SAFE(pathIntersectionsBuffer, compactedPathRayIdx));
    float3 planeNormalWS;
    float3 interpVertexNormalWS;
    GetNormalsAtRayIntersection(compactedPathRayIdx,
        instanceId,
        pathIntersectionsBuffer,
        instanceIdToMeshDataOffsets,
        instanceIdToInvTransposedMatrices,
        geometryPositionsBuffer,
        geometryNormalsBuffer,
        geometryIndicesBuffer,
        &planeNormalWS,
        &interpVertexNormalWS
        KERNEL_VALIDATOR_BUFFERS);

    unsigned char isNormalFacingTheRay = 1;
    const bool frontFacing = dot(planeNormalWS, INDEX_SAFE(pathRaysBuffer_0, compactedPathRayIdx).d.xyz) <= 0.0f;
    if (!frontFacing)
    {
        const MaterialTextureProperties matProperty = INDEX_SAFE(instanceIdToTransmissionTextureProperties, instanceId);
        bool isDoubleSidedGI = GetMaterialProperty(matProperty, kMaterialInstanceProperties_DoubleSidedGI);
        planeNormalWS =        isDoubleSidedGI ? -planeNormalWS        : planeNormalWS;
        interpVertexNormalWS = isDoubleSidedGI ? -interpVertexNormalWS : interpVertexNormalWS;
        isNormalFacingTheRay = isDoubleSidedGI? 1 : 0;
        if (validitybufferMode == ValidityBufferMode_Generate && !isDoubleSidedGI)
        {
            const int expandedPathRayIdx = Ray_GetExpandedIndex(GET_PTR_SAFE(pathRaysBuffer_0, compactedPathRayIdx));
            //We use the shadowmaskExpandedBuffer.y to store validity to avoid having an additional expanded buffer.
            INDEX_SAFE(shadowmaskExpandedBuffer, expandedPathRayIdx).y = 1.0f;
        }
    }

    // Store normals for various kernels to use later
    INDEX_SAFE(pathLastPlaneNormalBuffer, compactedPathRayIdx) = EncodeNormalTo888(planeNormalWS);
    INDEX_SAFE(pathLastInterpNormalBuffer, compactedPathRayIdx) = EncodeNormalTo888(interpVertexNormalWS);
    INDEX_SAFE(pathLastNormalFacingTheRayBuffer, compactedPathRayIdx) = isNormalFacingTheRay;
}
