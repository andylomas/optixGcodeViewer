#pragma once

#include "ofMain.h"

#ifndef USING_OPTIX
using namespace glm;

#define float3 vec3
#define make_float3 vec3
#endif

enum SlicerType { slicer_none, slicer_cura, slicer_simplify3D, slicer_prusaSlicer };

struct PathSegment {
	std::vector<float3> positions;
	std::vector<float> extrusions;
	float layerThickness = 0;
	float nozzleWidth = 0;
};

class GcodeReader {
public:
	void readPathsFromGcode(std::string fileName);
	void printPaths();
	void cleanupPaths(float minSegmentLength = 0.1f);
	void convertPathsToLines();
	void convertPathsToMesh(float nozzleWidth, float layerThickness, int tubeResolution, int endCapResolution, float maxAngle=45);
	float3 getMidPos();
	ofMesh &getMesh() { return myMesh; }
	void draw();

private:
	void clearPaths();
	void addCrossSection(float3 p, float3 d, float scaleFactor, float nozzleWidth, float layerThickness, int tubeResolution);
	void addEndCap(float3 p, float3 d, float nozzleWidth, float layerThickness, int tubeResolution, int endCapResolution);

	std::vector<PathSegment> myPaths;
	ofMesh myMesh;
	SlicerType mySlicer = slicer_none;
};
