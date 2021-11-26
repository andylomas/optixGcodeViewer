#include "ofMain.h"


//--------------------------------------------------------------
void ofMesh::clear() {
    myVertices.clear();
    myNormals.clear();
    myTriangleIndices.clear();
}

//--------------------------------------------------------------
void ofMesh::addVertex( float3 p ) {
    myVertices.push_back(p);
}

//--------------------------------------------------------------
void ofMesh::addNormal( float3 n) {
    myNormals.push_back(n);
}

//--------------------------------------------------------------
void ofMesh::addTriangle( unsigned int i0, unsigned int i1, unsigned int i2 )  {
    myTriangleIndices.push_back(make_uint3(i0, i1, i2));
}

//--------------------------------------------------------------
int ofMesh::getRange(float3 &rangeMin, float3 &rangeMax) {
    // Quick exit if no vertices
    if (myVertices.size() == 0)
    {
        rangeMin = make_float3(0, 0, 0);
        rangeMin = make_float3(0, 0, 0);
        return -1;
    }

    rangeMin = myVertices[0];
    rangeMax = myVertices[0];

    for (unsigned int i = 1; i < myVertices.size(); i++) {
        float3 v = myVertices[i];
        if (v.x < rangeMin.x) rangeMin.x = v.x;
        if (v.y < rangeMin.y) rangeMin.y = v.y;
        if (v.z < rangeMin.z) rangeMin.z = v.z;
        if (v.x > rangeMax.x) rangeMax.x = v.x;
        if (v.y > rangeMax.y) rangeMax.y = v.y;
        if (v.z > rangeMax.z) rangeMax.z = v.z;
    }

    return 0;
}

//--------------------------------------------------------------
float3 ofMesh::getMidPos() {
    // Quick exit if no vertices
    if (myVertices.size() == 0)
    {
        return make_float3(0, 0, 0);
    }

    float3 rangeMin, rangeMax;
    getRange(rangeMin, rangeMax);
    return 0.5 * (rangeMin + rangeMax);
}

//--------------------------------------------------------------
void ofMesh::fitToBounds(float3 boundMin, float3 boundMax) {
    float3 rangeMin, rangeMax;
    getRange(rangeMin, rangeMax);

    float3 scaleFactors = (boundMax - boundMin) / (rangeMax - rangeMin);
    float minScaleFactor = scaleFactors.x;
    if (scaleFactors.y < minScaleFactor) minScaleFactor = scaleFactors.y;
    if (scaleFactors.z < minScaleFactor) minScaleFactor = scaleFactors.z;

    float3 offset = 0.5f * (boundMin + boundMax) -0.5f * (rangeMin + rangeMax);

    for (unsigned int i = 0; i < myVertices.size(); i++) {
        myVertices[i] = minScaleFactor * (myVertices[i] + offset);
    }
}

//--------------------------------------------------------------
void ofMesh::flipYZ() {
    for (unsigned int i = 0; i < myVertices.size(); i++) {
        float3 v = myVertices[i];
        myVertices[i] = make_float3(v.x, v.z, -v.y);
    }

    for (unsigned int i = 0; i < myNormals.size(); i++) {
        float3 n = myNormals[i];
        myNormals[i] = make_float3(n.x, n.z, -n.y);
    }
}

//--------------------------------------------------------------
void ofMesh::readPlyFile( std::string filePath ) {
    // Clear any existing mesh data
    clear();

    // Open text file and read data from it
    std::ifstream infile(filePath);

    if (!infile.is_open()) {
        std::cout << "Error: couldn't open file " << filePath << " for reading\n";
        return;
    }

    // Get header data from ply file a line at a time
    std::string curLine;
    std::string arg0, arg1, arg2;
    int vertex_property_x = -1;
    int vertex_property_y = -1;
    int vertex_property_z = -1;
    int vertex_property_nx = -1;
    int vertex_property_ny = -1;
    int vertex_property_nz = -1;
    int property_index = -1;
    int num_vertex_properties = -1;
    unsigned int numVertices;
    unsigned int numFaces;
    while (getline(infile, curLine)) {

        // Create stream from the current line
        std::istringstream ss(curLine);

        // Get the first argument
        ss >> arg0;

        if (arg0 == "end_header") {
            break;
        }

        if (arg0 == "element") {
            ss >> arg1;

            if (arg1 == "vertex") {
                ss >> numVertices;
                property_index = 0;
            }

            if (arg1 == "face") {
                ss >> numFaces;
            }
        }

        if (arg0 == "property") {
            ss >> arg1;

            if (arg1 == "float") {
                ss >> arg2;
                if (arg2 == "x") {
                    vertex_property_x = property_index;
                    num_vertex_properties = property_index + 1;
                }

                if (arg2 == "y") {
                    vertex_property_y = property_index;
                    num_vertex_properties = property_index + 1;
                }

                if (arg2 == "z") {
                    vertex_property_z = property_index;
                    num_vertex_properties = property_index + 1;
                }

                if (arg2 == "nx") {
                    vertex_property_nx = property_index;
                    num_vertex_properties = property_index + 1;
                }

                if (arg2 == "ny") {
                    vertex_property_ny = property_index;
                    num_vertex_properties = property_index + 1;
                }

                if (arg2 == "nz") {
                    vertex_property_nz = property_index;
                    num_vertex_properties = property_index + 1;
                }

                property_index++;
            }
        }
    }

    std::cout << "numVertices: " << numVertices << "\n";
    std::cout << "numFaces: " << numFaces << "\n";
    std::cout << "num_vertex_properties: " << num_vertex_properties << "\n";
    std::cout << "vertex_property x, y, z: " << vertex_property_x << ", " << vertex_property_y << ", " << vertex_property_z << "\n";
    std::cout << "vertex_property nx, ny, nz: " << vertex_property_nx << ", " << vertex_property_ny << ", " << vertex_property_nz << "\n";

    // Reserve space in the vectors
    myVertices.reserve(numVertices);
    myTriangleIndices.reserve(numFaces);
    if (vertex_property_nx > 0)
        myNormals.reserve(numVertices);

    // Read vertex data
    float x = 0, y = 0, z = 0;
    float nx = 0, ny = 0, nz = 0;
    for (unsigned int i = 0; i < numVertices; i++) {
        if (getline(infile, curLine)) {
            // Create stream from the current line
            std::istringstream ss(curLine);

            // Read the vertex property values
            float val;
            for (int j = 0; j < num_vertex_properties; j++) {
                ss >> val;
                if (j == vertex_property_x) x = val;
                if (j == vertex_property_y) y = val;
                if (j == vertex_property_z) z = val;
                if (j == vertex_property_nx) nx = val;
                if (j == vertex_property_ny) ny = val;
                if (j == vertex_property_nz) nz = val;
            }

            float3 p = make_float3(x, y, z);
            if (p.x == 0 || p.y == 0 || p.z == 0) {
                std::cout << "curLine: " << curLine << "\n";
                std::cout << p.x << ", " << p.y << ", " << p.z << "\n";
            }
            // Set vertex position
            myVertices.push_back(make_float3(x, y, z));

            // Set normal
            if (vertex_property_nx > 0) {
                myNormals.push_back(make_float3(nx, ny, nz));
            }
        }
        else {
            std::cout << "Error: couldn't read data for vertex " << i << "\n";
        }
    }

    // Read face data
    int i0, i1, i2;
    for (unsigned int i = 0; i < numFaces; i++) {
        if (getline(infile, curLine)) {
            // Create stream from the current line
            std::istringstream ss(curLine);

            // Read the face property values
            int face_size;
            ss >> face_size;

            for (int j = 0; j < face_size; j++) {
                if (j == 0) ss >> i0;

                if (j == 1) ss >> i1;

                if (j > 1) {
                    ss >> i2;

                    // Set triangle using the current vertex indices
                    myTriangleIndices.push_back(make_uint3(i0, i1, i2));

                    // To handle a face with multiple triangles simply make i1 = i2 after setting the current triangle
                    i1 = i2;
                }
            }
        }
        else {
            std::cout << "Error: couldn't read data for face " << i << "\n";
        }
    }
}