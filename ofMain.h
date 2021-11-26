#pragma once

// Minimal openFrameworks function declarations for compatability

#include <cuda_runtime.h>
#include <vector>
#include <sutil/vec_math.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <assert.h>
#undef NDEBUG

#define USING_OPTIX

#define PI 3.1415927f
#define TWO_PI 6.2831853f

enum ofMeshDislayMode { OF_PRIMITIVE_LINES, OF_PRIMITIVE_TRIANGLES };

class ofMesh {
public:
    void clear();
    void addVertex( float3 p );
    void addNormal( float3 n );
    void addTriangle( unsigned int i0, unsigned int i1, unsigned int i2 );
    std::vector<float3> &getVertices() { return myVertices; }
    std::vector<float3> &getNormals() { return myNormals; }
    std::vector<uint3> &getTriangleIndices() { return myTriangleIndices; }
    float3 *getVertexData() { return myVertices.data(); }
    float3 *getNormalData() { return myNormals.data(); }
    uint3 *getTriangleIndexData() { return myTriangleIndices.data(); }
    unsigned int getNumVertices() { return myVertices.size(); }
    unsigned int getNumNormals() { return myNormals.size(); }
    unsigned int getNumTriangles() { return myTriangleIndices.size(); }
    int getRange(float3 &rangeMin, float3 &rangeMax);
    float3 getMidPos();
    void fitToBounds(float3 boundMin = make_float3(-1, -1, -1), float3 boundMax = make_float3(1, 1, 1));
    void flipYZ();
    void readPlyFile( std::string filePath );
    void draw() {}
    void setMode( ofMeshDislayMode curMode ) {};

private:
    std::vector<float3> myVertices;
    std::vector<float3> myNormals;
    std::vector<uint3> myTriangleIndices;
};
