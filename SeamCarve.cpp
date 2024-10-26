#include <opencv2/opencv.hpp>  // Include OpenCV header
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>

using namespace std;
using namespace cv;  // OpenCV namespace

// Calculate the energy map using the gradient magnitude
vector<vector<int>> calculateEnergyMap(unsigned char* image, int width, int height, int channels) {
    vector<vector<int>> energyMap(height, vector<int>(width, 0));

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int offset = (y * width + x) * channels;

            int dxR = image[offset + channels] - image[offset - channels];
            int dyR = image[offset + width * channels] - image[offset - width * channels];
            int dxG = image[offset + channels + 1] - image[offset - channels + 1];
            int dyG = image[offset + width * channels + 1] - image[offset - width * channels + 1];
            int dxB = image[offset + channels + 2] - image[offset - channels + 2];
            int dyB = image[offset + width * channels + 2] - image[offset - width * channels + 2];

            int energy = sqrt(dxR * dxR + dyR * dyR + dxG * dxG + dyG * dyG + dxB * dxB + dyB * dyB);
            energyMap[y][x] = energy;
        }
    }

    return energyMap;
}

// Find the optimal vertical seam using dynamic programming
vector<int> findVerticalSeam(const vector<vector<int>>& energyMap) {
    int height = energyMap.size();
    int width = energyMap[0].size();

    vector<vector<int>> seamEnergy = energyMap;
    vector<int> seam(height);

    for (int y = 1; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int minEnergy = seamEnergy[y - 1][x];
            if (x > 0) minEnergy = min(minEnergy, seamEnergy[y - 1][x - 1]);
            if (x < width - 1) minEnergy = min(minEnergy, seamEnergy[y - 1][x + 1]);

            seamEnergy[y][x] += minEnergy;
        }
    }

    int minSeamEnd = min_element(seamEnergy[height - 1].begin(), seamEnergy[height - 1].end()) - seamEnergy[height - 1].begin();
    seam[height - 1] = minSeamEnd;

    for (int y = height - 2; y >= 0; --y) {
        int x = seam[y + 1];
        int minX = x;

        if (x > 0 && seamEnergy[y][x - 1] < seamEnergy[y][minX]) minX = x - 1;
        if (x < width - 1 && seamEnergy[y][x + 1] < seamEnergy[y][minX]) minX = x + 1;

        seam[y] = minX;
    }

    return seam;
}

// Find the vertical seam using a greedy approach (selects local minimum energy per row)
vector<int> findVerticalSeamGreedy(const vector<vector<int>>& energyMap) {
    int height = energyMap.size();
    int width = energyMap[0].size();
    vector<int> seam(height);

    // Start from the minimum energy position in the top row
    seam[0] = min_element(energyMap[0].begin(), energyMap[0].end()) - energyMap[0].begin();

    for (int y = 1; y < height; ++y) {
        int x = seam[y - 1];
        int minX = x;

        // Look at the current row and pick the local minimum from three possible positions
        if (x > 0 && energyMap[y][x - 1] < energyMap[y][minX]) minX = x - 1;
        if (x < width - 1 && energyMap[y][x + 1] < energyMap[y][minX]) minX = x + 1;

        seam[y] = minX;
    }

    return seam;
}

// Remove the seam from the image
void removeVerticalSeam(unsigned char*& image, int& width, int height, int channels, const vector<int>& seam) {
    unsigned char* newImage = new unsigned char[width * height * channels - height * channels];

    for (int y = 0; y < height; ++y) {
        int newIndex = y * (width - 1) * channels;
        int oldIndex = y * width * channels;

        for (int x = 0; x < width; ++x) {
            if (x != seam[y]) {
                for (int c = 0; c < channels; ++c) {
                    newImage[newIndex++] = image[oldIndex + x * channels + c];
                }
            }
        }
    }

    delete[] image;
    image = newImage;
    --width;
}

// Helper function to transpose the image matrix
void transposeImage(unsigned char*& image, int& width, int& height, int channels) {
    unsigned char* transposedImage = new unsigned char[width * height * channels];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                transposedImage[(x * height + y) * channels + c] = image[(y * width + x) * channels + c];
            }
        }
    }
    swap(width, height);
    delete[] image;
    image = transposedImage;
}

// Display the image using OpenCV
void displayImage(const unsigned char* image, int width, int height, int channels, const string& windowName) {
    Mat mat(height, width, channels == 3 ? CV_8UC3 : CV_8UC1, (void*)image);
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, mat);
    waitKey(0);  // Display briefly and return control to the console
}

int main() {
    string filename;
    cout << "Enter the base name of the image (e.g., 'surfer'): ";
    cin >> filename;
    filename += ".png";  // Append .png to the filename

    // Load the image using OpenCV
    Mat img = imread(filename, IMREAD_COLOR);
    if (img.empty()) {
        cout << "Failed to load image" << endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    unsigned char* image = img.data;

    // Display the original image briefly
    displayImage(image, width, height, channels, "Original Image");

    while (true) {
        string input;
        int targetWidth, targetHeight;

        while (true) {
            cout << "Enter the target width and height (less than " << width << " " << height << "), or type '-1' to exit: ";
            getline(cin >> ws, input);

            if (input == "-1") return 0;  // Exit condition

            istringstream iss(input);
            if (iss >> targetWidth >> targetHeight && targetWidth < width && targetHeight < height) {
                break;  // Valid input, break out of the inner loop
            }
            else {
                cout << "Invalid input. Please enter two integers less than " << width << " and " << height << "." << endl;
            }
        }

        // Resize using dynamic programming
        int dpWidth = width, dpHeight = height;
        unsigned char* dpImage = new unsigned char[dpWidth * dpHeight * channels];
        memcpy(dpImage, image, dpWidth * dpHeight * channels);

        // Reduce width using dynamic programming
        while (dpWidth > targetWidth) {
            auto energyMap = calculateEnergyMap(dpImage, dpWidth, dpHeight, channels);
            auto seam = findVerticalSeam(energyMap);  // Dynamic programming
            removeVerticalSeam(dpImage, dpWidth, dpHeight, channels, seam);
        }

        // Reduce height by transposing for vertical seam removal
        if (dpHeight > targetHeight) {
            transposeImage(dpImage, dpWidth, dpHeight, channels);  // Transpose for seam removal along new "width"
            while (dpWidth > targetHeight) {
                auto energyMap = calculateEnergyMap(dpImage, dpWidth, dpHeight, channels);
                auto seam = findVerticalSeam(energyMap);  // Dynamic programming
                removeVerticalSeam(dpImage, dpWidth, dpHeight, channels, seam);
            }
            transposeImage(dpImage, dpWidth, dpHeight, channels);  // Transpose back to original orientation
        }

        // Resize using greedy approach
        int greedyWidth = width, greedyHeight = height;
        unsigned char* greedyImage = new unsigned char[greedyWidth * greedyHeight * channels];
        memcpy(greedyImage, image, greedyWidth * greedyHeight * channels);

        // Reduce width using greedy approach
        while (greedyWidth > targetWidth) {
            auto energyMap = calculateEnergyMap(greedyImage, greedyWidth, greedyHeight, channels);
            auto seam = findVerticalSeamGreedy(energyMap);  // Greedy approach
            removeVerticalSeam(greedyImage, greedyWidth, greedyHeight, channels, seam);
        }

        // Reduce height using greedy approach
        if (greedyHeight > targetHeight) {
            transposeImage(greedyImage, greedyWidth, greedyHeight, channels);  // Transpose for seam removal along new "width"
            while (greedyWidth > targetHeight) {
                auto energyMap = calculateEnergyMap(greedyImage, greedyWidth, greedyHeight, channels);
                auto seam = findVerticalSeamGreedy(energyMap);  // Greedy approach
                removeVerticalSeam(greedyImage, greedyWidth, greedyHeight, channels, seam);
            }
            transposeImage(greedyImage, greedyWidth, greedyHeight, channels);  // Transpose back to original orientation
        }

        // Display the resized images
        displayImage(dpImage, dpWidth, dpHeight, channels, "Dynamic Programming Resized Image");
        displayImage(greedyImage, greedyWidth, greedyHeight, channels, "Greedy Resized Image");

        // Save both resized images
        Mat dpMat(dpHeight, dpWidth, channels == 3 ? CV_8UC3 : CV_8UC1, dpImage);
        Mat greedyMat(greedyHeight, greedyWidth, channels == 3 ? CV_8UC3 : CV_8UC1, greedyImage);
        imwrite("dp_resized.png", dpMat);
        imwrite("greedy_resized.png", greedyMat);
        cout << "Dynamic programming resized image saved as dp_resized.png" << endl;
        cout << "Greedy resized image saved as greedy_resized.png" << endl;

        // Free memory
        delete[] dpImage;
        delete[] greedyImage;
    }

    cout << "Program terminated." << endl;
    return 0;
}


