#pragma once

inline __device__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}


inline __device__ float log_lerp(float a, float k, float t)
{
    return a * expf(k * t);
}


// calculates index from grid position, wrapping if necessary
__device__ __forceinline__
uint calcIndex(uint px, uint py, uint pz, uint3 gridSize, uint3 gridSizeMask, bool rdeMirrorMode)
{
	// Clamp px, py and pz to the relevant range
    uint ppx = px & gridSizeMask.x;
    uint ppy = py & gridSizeMask.y;
    uint ppz = pz & gridSizeMask.z;

    if (rdeMirrorMode)
    {
        // If we're in mirror mode, flip the values if we're in an odd numbered repetition
        if (px & gridSize.x) ppx = gridSizeMask.x - ppx;
        if (py & gridSize.y) ppy = gridSizeMask.y - ppy;
        if (pz & gridSize.z) ppz = gridSizeMask.z - ppz;
    }

    return (ppz*gridSize.x*gridSize.y) + (ppy*gridSize.x) + ppx;
}


// compute position in 3d grid from 1d index
// only works for power of 2 sizes
__device__ __forceinline__
uint3 calcGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
{
    uint3 gridPos;
    gridPos.x = i & gridSizeMask.x;
    gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
    gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;
    return gridPos;
}

__device__ __forceinline__
uchar floatToUchar(float v)
{
	return v < 0 ? 0 : ((v > 1) ? 255 : 255.f * v);
}


__device__ __forceinline__
uint16_t floatToUint16(float v)
{
	return v < 0 ? 0 : ((v > 1) ? 65535 : 65535.f * v);
}


__device__ __forceinline__
float ucharToFloat(uchar v)
{
    return v / 255.f;
}


__device__ __forceinline__
float uint16ToFloat(uint16_t v)
{
    return v / 65535.f;
}
