#ifndef ENVIRONMENT_LIGHTING_H
#define ENVIRONMENT_LIGHTING_H

#include "commonCL.h"

enum
{
    FACE_POS_X = 0,
    FACE_NEG_X,
    FACE_POS_Y,
    FACE_NEG_Y,
    FACE_POS_Z,
    FACE_NEG_Z
};

float4 TextureSampleTrilinear(__global const float4 * const env_mipped_cube_texels_buffer, __global const int * const env_mip_offsets_buffer, int face_idx, int dim_face, int nrMips, const float Sin, const float Tin, const float fLodIn KERNEL_VALIDATOR_BUFFERS_DEF);

static float4 EnvCubeMapSample(__global const float4 * const env_mipped_cube_texels_buffer, __global const int * const env_mip_offsets_buffer,
    int dim_face, int nrMips, float3 dir, float lod KERNEL_VALIDATOR_BUFFERS_DEF)
{
    const float vx = dir.x, vy = dir.y, vz = dir.z;

    int NumMaxCoordIdx = 0;
    float NumMaxCoord = vx;
    if (fabs(vz) >= fabs(vx) && fabs(vz) >= fabs(vy))
    {
        NumMaxCoordIdx = 2;
        NumMaxCoord = vz;
    }
    else if (fabs(vy) > fabs(vx))
    {
        NumMaxCoordIdx = 1;
        NumMaxCoord = vy;
    }

    bool IsPosSign = NumMaxCoord >= 0.0f;

    float S, T;

    int f = 0;
    switch (NumMaxCoordIdx)
    {
        case 0:
            f = IsPosSign ? FACE_POS_X : FACE_NEG_X;
            S = (IsPosSign ? (-1.0f) : 1.0f) * vz;
            T = vy;
            break;
        case 1:
            f = IsPosSign ? FACE_POS_Y : FACE_NEG_Y;
            S = vx;
            T = (IsPosSign ? (-1.0f) : 1.0f) * vz;
            break;
        default:
            f = IsPosSign ? FACE_POS_Z : FACE_NEG_Z;
            S = (IsPosSign ? 1.0f : (-1.0f)) * vx;
            T = vy;
    }

    float Denom = fabs(NumMaxCoord);
    Denom = Denom < FLT_EPSILON ? FLT_EPSILON : Denom;
    S /= Denom; T /= Denom;


    return TextureSampleTrilinear(env_mipped_cube_texels_buffer, env_mip_offsets_buffer, f, dim_face, nrMips, 0.5f * S + 0.5f, 0.5f * (-T) + 0.5f, lod KERNEL_VALIDATOR_BUFFERS);
}

float4 TextureSampleBilinear(__global const float4 * const env_mipped_cube_texels_buffer, const int mipOffset, const int face_idx, const int mip_dim, const float S, const float T KERNEL_VALIDATOR_BUFFERS_DEF);

float4 TextureSampleTrilinear(__global const float4 * const env_mipped_cube_texels_buffer, __global const int * const env_mip_offsets_buffer,
    int face_idx, int dim_face, int nrMips, const float Sin, const float Tin, const float fLodIn KERNEL_VALIDATOR_BUFFERS_DEF)
{
    const float S = min(1.0f, max(0.0f, Sin));
    const float T = min(1.0f, max(0.0f, Tin));
    const float fLod = fLodIn > (nrMips - 1) ? (nrMips - 1) : (fLodIn < 0.0f ? 0.0f : fLodIn);
    const int iLod0 = (int)fLod;
    const int iLod1 = min(iLod0 + 1, nrMips - 1);
    const float Mix = fLod - iLod0;

    const float4 pix0 = TextureSampleBilinear(env_mipped_cube_texels_buffer, INDEX_SAFE(env_mip_offsets_buffer, iLod0), face_idx, dim_face >> iLod0, S, T KERNEL_VALIDATOR_BUFFERS);
    const float4 pix1 = TextureSampleBilinear(env_mipped_cube_texels_buffer, INDEX_SAFE(env_mip_offsets_buffer, iLod1), face_idx, dim_face >> iLod1, S, T KERNEL_VALIDATOR_BUFFERS);

    return (1 - Mix) * pix0 + Mix * pix1;
}

// generate an offset into a 2D image with a 1 pixel wide skirt.
static int GenImgExtIdx(const int x, const int y, const int dim)
{
    return (y + 1) * (dim + 2) + (x + 1);
}

// this function assumes S and T are in [0;1] and the presence of borderpixels/skirt with the texture
float4 TextureSampleBilinear(__global const float4 * const env_mipped_cube_texels_buffer, const int mipOffset, const int face_idx, const int mip_dim, const float S, const float T KERNEL_VALIDATOR_BUFFERS_DEF)
{
    const float U = S * mip_dim - 0.5f, V = T * mip_dim - 0.5f;
    const int offset = mipOffset + ((mip_dim + 2) * (mip_dim + 2) * face_idx);

    // technically same as a floorf() since we know -0.5f is the min. possible value for U and V
    const int u0 = ((int)(U + 1.0f)) - 1;
    const int v0 = ((int)(V + 1.0f)) - 1;

    const float dx = U - u0, dy = V - v0;
    const float weights[] = {(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy};

    float4 res = 0.0f;

    for (int y = 0; y < 2; y++)
    {
        for (int x = 0; x < 2; x++)
        {
            const int idx = GenImgExtIdx(u0 + x, v0 + y, mip_dim);

            const int weightsIdx = 2 * y + x;
            KERNEL_ASSERT(weightsIdx < 4);
            res += weights[weightsIdx] * INDEX_SAFE(env_mipped_cube_texels_buffer, offset + idx);
        }
    }

    return res;
}

static bool UseEnvironmentMIS(uint envFlags)
{
    return (envFlags & 1) != 0;
}

static bool SampleDirectEnvironment(uint envFlags)
{
    return (envFlags & 2) == 0;
}

static bool SampleIndirectEnvironment(uint envFlags)
{
    return (envFlags & 4) == 0;
}

static int GetRaysPerEnvironmentIndirect(const Environment environmentInputData)
{
    return environmentInputData.numRaysIndirect;
}

// MIS path related functions to estimate pdfs.
static float EnvironmentMetric(const float3 intensity)
{
    // use the max intensity as a metric. Keep in sync with EnvironmentMetric in PVRJobUpdateEnvironmentLighting.cpp and the RLSL version.
    return max(max(intensity.x, intensity.y), intensity.z);
}

// Calculates the MIS weight using a balanced heuristic.
static float EnvironmentHeuristic(const float pdf1, const float pdf2)
{
    const float denom = pdf1 + pdf2;
    return denom > 0.0f ? (pdf1 / denom) : 0.0f;
}

// Creates an mis direction for occlusion testing the environment. If the returned vector has .w == 0.0f the ray is inactive
float4 GenerateSurfaceEnvironmentRayMIS(const int numEnvironmentSamples, const float3 interpNormal, const float3 geomNormal, const float3 rand, INPUT_BUF(, PackedNormalOctQuad, envDirectionsBuffer)KERNEL_VALIDATOR_BUFFERS_DEF)
{
    float4 dir;
    if (rand.z > 0.5f)
    {
        float floored;
        int sampleIndex = ((int)(fract(rand.x, &floored) * (float)numEnvironmentSamples)) % numEnvironmentSamples;
        dir.xyz = DecodeNormal(INDEX_SAFE(envDirectionsBuffer, sampleIndex));
        dir.w   = -1.0f;
    }
    else
    {
        dir.xyz = Sample_MapToHemisphere(rand.xy, interpNormal, 1.f); // last param of 1.f makes the directions cosine distributed
        dir.w   = 1.0f;
    }
    float cosdir   = fmax(0.0f, dot(dir.xyz, interpNormal));
    bool  shootRay = dot(dir.xyz, geomNormal) > 0.0f && cosdir > 0.0f;
    dir.w    = shootRay ? dir.w : 0.0f;

    return dir;
}

// Creates a cosine distributed random direction for occlusion testing the environment.
float4 GenerateSurfaceEnvironmentRay(const float3 interpNormal, const float3 geomNormal, const float2 rand)
{
    float4 dir;
    dir.xyz = Sample_MapToHemisphere(rand.xy, interpNormal, 1.f);               // last param of 1.f makes the directions cosine distributed
    float cosdir = fmax(0.0f, dot(dir.xyz, interpNormal));
    bool shootRay = dot(dir.xyz, geomNormal) > 0.0f && cosdir > 0.0f;
    dir.w = shootRay ? 1.0f : 0.0f;

    return dir;
}

float4 GenerateVolumeEnvironmentRayMIS(const int numEnvironmentSamples, float3 rand, INPUT_BUF(, PackedNormalOctQuad, envDirectionsBuffer)KERNEL_VALIDATOR_BUFFERS_DEF)
{
    float4 dir;
    if (rand.z > 0.5f)
    {
        float floored;
        int sampleIndex = ((int)(fract(rand.x, &floored) * (float)numEnvironmentSamples)) % numEnvironmentSamples;
        dir.xyz = DecodeNormal(INDEX_SAFE(envDirectionsBuffer, sampleIndex));
        dir.w   = -1.0f;
    }
    else
    {
        dir.xyz = Sample_MapToSphere(rand.xy);
        dir.w   = 1.0f;
    }
    return dir;
}

float4 GenerateVolumeEnvironmentRay(float2 rand)
{
    float4 dir;
    dir.xyz = Sample_MapToSphere(rand);
    dir.w   = 1.0f;

    return dir;
}

// Returns the environment color for the given direction when using mis sampling.
float3 ProcessEnvironmentRayMIS(float4 direction, float3 interpNormal, int envDim, int envNumMips, float envIntegral,
    INPUT_BUF(, float4, env_mipped_cube_texels_buffer), INPUT_BUF(, int, env_mip_offsets_buffer)KERNEL_VALIDATOR_BUFFERS_DEF)
{
    bool   useEnv      = direction.w < 0.0f; // the generate functions encode the sign. w == 0.0f can be ignored, as inactive rays should never get this far
    float  cosdir      = fmax(0.0f, dot(direction.xyz, interpNormal));
    float  pdf_diffuse = cosdir / PI;
    float3 intensity   = EnvCubeMapSample(env_mipped_cube_texels_buffer, env_mip_offsets_buffer, envDim, envNumMips, direction.xyz, 0.0f KERNEL_VALIDATOR_BUFFERS).xyz;
    float  metric      = EnvironmentMetric(intensity);
    float  pdf_envmap  = metric / envIntegral;
    float  mis_weight  = useEnv ? EnvironmentHeuristic(pdf_envmap, pdf_diffuse) : EnvironmentHeuristic(pdf_diffuse, pdf_envmap);
    // Use one random sample rule instead of estimating both pdfs.
    // Due to this the chosen path has its mis weight multiplied by 2.0 as we're evenly drawing from the two pdfs.
    float  weight      = 2.0f * mis_weight * (useEnv ? (cosdir / PI / pdf_envmap) : 1.0f); // non-mis is 1.0 because cosdir / lambert_PI / (pdf_diffuse) = 1.0

    return intensity * weight;
}

// Returns the environment color for the given direction without using mis sampling.
float3 ProcessEnvironmentRay(float4 direction, int envDim, int envNumMips,
    INPUT_BUF(, float4, env_mipped_cube_texels_buffer), INPUT_BUF(, int, env_mip_offsets_buffer)KERNEL_VALIDATOR_BUFFERS_DEF)
{
    float3 intensity = EnvCubeMapSample(env_mipped_cube_texels_buffer, env_mip_offsets_buffer, envDim, envNumMips, direction.xyz, 0.0f KERNEL_VALIDATOR_BUFFERS).xyz;
    return intensity;
}

float3 ProcessVolumeEnvironmentRayMIS(float4 direction, int envDim, int envNumMips, float envIntegral,
    INPUT_BUF(, float4, env_mipped_cube_texels_buffer), INPUT_BUF(, int, env_mip_offsets_buffer)KERNEL_VALIDATOR_BUFFERS_DEF)
{
    bool   useEnv      = direction.w < 0.0f; // the generate functions encode the sign. w == 0.0f can be ignored, as inactive rays should never get this far
    float  pdf_diffuse = 1.0f / (4.0f * PI);
    float3 intensity   = EnvCubeMapSample(env_mipped_cube_texels_buffer, env_mip_offsets_buffer, envDim, envNumMips, direction.xyz, 0.0f KERNEL_VALIDATOR_BUFFERS).xyz;
    float  metric      = EnvironmentMetric(intensity);
    float  pdf_envmap  = metric / envIntegral;
    float  mis_weight  = useEnv ? EnvironmentHeuristic(pdf_envmap, pdf_diffuse) : EnvironmentHeuristic(pdf_diffuse, pdf_envmap);
    // Use one random sample rule instead of estimating both pdfs.
    // Due to this the chosen path has its mis weight multiplied by 2.0 as we're evenly drawing from the two pdfs.
    float  weight      = 2.0f * mis_weight * (useEnv ? (1.0f / (PI * pdf_envmap)) : 4.0f);

    return intensity * weight;
}

float3 ProcessVolumeEnvironmentRay(float4 direction, int envDim, int envNumMips,
    INPUT_BUF(, float4, env_mipped_cube_texels_buffer), INPUT_BUF(, int, env_mip_offsets_buffer)KERNEL_VALIDATOR_BUFFERS_DEF)
{
    float3 intensity = EnvCubeMapSample(env_mipped_cube_texels_buffer, env_mip_offsets_buffer, envDim, envNumMips, direction.xyz, 0.0f KERNEL_VALIDATOR_BUFFERS).xyz;
    float  pdf_diffuse = 1.0 / 4.0; // PI in denominator cancels out with SH
    return intensity / pdf_diffuse;
}

#endif // ENVIRONMENT_LIGHTING_H
