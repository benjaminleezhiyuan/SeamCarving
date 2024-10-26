#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Function to compute the energy map of the image
Mat computeEnergyMap(const Mat& image) {
    Mat gray, grad_x, grad_y, abs_grad_x, abs_grad_y, energy_map;

    // Convert to grayscale
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Compute gradients along X and Y directions
    Sobel(gray, grad_x, CV_16S, 1, 0, 3);
    Sobel(gray, grad_y, CV_16S, 0, 1, 3);

    // Convert gradients to absolute values
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    // Compute the energy map as the sum of absolute gradients
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, energy_map);

    return energy_map;
}

// Function to find and remove a vertical seam using dynamic programming
void removeVerticalSeamDP(Mat& image) {
    Mat energy_map = computeEnergyMap(image);
    int rows = energy_map.rows;
    int cols = energy_map.cols;

    // Initialize the cumulative energy map
    Mat M = Mat::zeros(rows, cols, CV_32S);
    energy_map.row(0).convertTo(M.row(0), CV_32S);

    // Compute the cumulative energy map
    for (int i = 1; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int min_energy = M.at<int>(i - 1, j);
            if (j > 0)
                min_energy = min(min_energy, M.at<int>(i - 1, j - 1));
            if (j < cols - 1)
                min_energy = min(min_energy, M.at<int>(i - 1, j + 1));
            M.at<int>(i, j) = energy_map.at<uchar>(i, j) + min_energy;
        }
    }

    // Backtrack to find the seam path
    vector<int> seam(rows);
    int min_idx = 0;
    int min_val = M.at<int>(rows - 1, 0);
    for (int j = 1; j < cols; j++) {
        if (M.at<int>(rows - 1, j) < min_val) {
            min_val = M.at<int>(rows - 1, j);
            min_idx = j;
        }
    }
    seam[rows - 1] = min_idx;

    for (int i = rows - 2; i >= 0; i--) {
        int prev_x = seam[i + 1];
        int min_energy = M.at<int>(i, prev_x);
        min_idx = prev_x;

        if (prev_x > 0 && M.at<int>(i, prev_x - 1) < min_energy) {
            min_energy = M.at<int>(i, prev_x - 1);
            min_idx = prev_x - 1;
        }
        if (prev_x < cols - 1 && M.at<int>(i, prev_x + 1) < min_energy) {
            min_energy = M.at<int>(i, prev_x + 1);
            min_idx = prev_x + 1;
        }
        seam[i] = min_idx;
    }

    // Remove the seam from the image
    Mat output(rows, cols - 1, CV_8UC3);
    for (int i = 0; i < rows; i++) {
        int idx = seam[i];
        for (int j = 0; j < idx; j++) {
            output.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
        }
        for (int j = idx + 1; j < cols; j++) {
            output.at<Vec3b>(i, j - 1) = image.at<Vec3b>(i, j);
        }
    }
    image = output.clone();
}

// Function to find and remove a horizontal seam using dynamic programming
void removeHorizontalSeamDP(Mat& image) {
    // Transpose the image to reuse the vertical seam function
    Mat transposed_image;
    transpose(image, transposed_image);
    flip(transposed_image, transposed_image, 0);

    removeVerticalSeamDP(transposed_image);

    // Transpose back to get the image with the horizontal seam removed
    flip(transposed_image, transposed_image, 0);
    transpose(transposed_image, image);
}

// Function to find and remove a vertical seam using a greedy algorithm
void removeVerticalSeamGreedy(Mat& image) {
    Mat energy_map = computeEnergyMap(image);
    int rows = energy_map.rows;
    int cols = energy_map.cols;

    // Initialize seam path
    vector<int> seam(rows);

    // Start from the top row
    double min_val;
    Point min_loc;
    minMaxLoc(energy_map.row(0), &min_val, nullptr, &min_loc, nullptr);
    seam[0] = min_loc.x;

    // Greedy approach to find the seam
    for (int i = 1; i < rows; i++) {
        int prev_x = seam[i - 1];
        int min_energy = energy_map.at<uchar>(i, prev_x);
        int min_idx = prev_x;

        if (prev_x > 0 && energy_map.at<uchar>(i, prev_x - 1) < min_energy) {
            min_energy = energy_map.at<uchar>(i, prev_x - 1);
            min_idx = prev_x - 1;
        }
        if (prev_x < cols - 1 && energy_map.at<uchar>(i, prev_x + 1) < min_energy) {
            min_energy = energy_map.at<uchar>(i, prev_x + 1);
            min_idx = prev_x + 1;
        }
        seam[i] = min_idx;
    }

    // Remove the seam from the image
    Mat output(rows, cols - 1, CV_8UC3);
    for (int i = 0; i < rows; i++) {
        int idx = seam[i];
        for (int j = 0; j < idx; j++) {
            output.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
        }
        for (int j = idx + 1; j < cols; j++) {
            output.at<Vec3b>(i, j - 1) = image.at<Vec3b>(i, j);
        }
    }
    image = output.clone();
}

// Function to find and remove a horizontal seam using a greedy algorithm
void removeHorizontalSeamGreedy(Mat& image) {
    // Transpose the image to reuse the vertical seam function
    Mat transposed_image;
    transpose(image, transposed_image);
    flip(transposed_image, transposed_image, 0);

    removeVerticalSeamGreedy(transposed_image);

    // Transpose back to get the image with the horizontal seam removed
    flip(transposed_image, transposed_image, 0);
    transpose(transposed_image, image);
}

#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    string filename;
    Mat original_image;

    // Loop to ensure a valid image file is loaded
    while (true) {
        cout << "Enter the image name (without extension): ";
        cin >> filename;
        cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Clear the newline character

        // Append .png to the filename
        filename += ".png";

        // Load the image
        original_image = imread(filename);
        if (!original_image.empty()) {
            break; // Exit the loop if a valid image is loaded
        }

        // If loading fails, display an error and prompt again
        cout << "Could not open or find the image! Please try again." << endl;
    }

    // Store the original image dimensions
    int original_width = original_image.cols;
    int original_height = original_image.rows;
    cout << "Original image dimensions: " << original_width << " x " << original_height << endl;

    while (true) {
        int new_width = -1, new_height = -1;
        cout << "Enter the desired new width and height (e.g., 500 500), 'new' to load a new image, or '-1' to exit: ";
        string input;
        getline(cin, input);

        if (input == "-1") {
            break; // Exit the loop if -1 is entered
        }
        else if (input == "new") {
            // Inner loop to load a new image if the user enters "new"
            while (true) {
                cout << "Enter the new image name (without extension): ";
                cin >> filename;
                cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Clear the newline character

                // Append .png to the filename
                filename += ".png";

                // Load the new image
                original_image = imread(filename);
                if (!original_image.empty()) {
                    break; // Exit the loop if a valid image is loaded
                }

                // If loading fails, display an error and prompt again
                cout << "Could not open or find the image! Please try again." << endl;
            }

            // Update the original image dimensions
            original_width = original_image.cols;
            original_height = original_image.rows;
            cout << "Original image dimensions: " << original_width << " x " << original_height << endl;
            continue; // Go back to the beginning of the loop
        }

        // Use a stringstream to parse the input
        stringstream ss(input);
        int width, height;
        if (!(ss >> width >> height) || (ss >> ws, !ss.eof())) {
            cout << "Invalid input. Please enter exactly two integer values for width and height." << endl;
            continue; // Prompt again if input is invalid
        }

        // Check that the dimensions are within bounds
        if (width <= 0 || width > original_width || height <= 0 || height > original_height) {
            cout << "Invalid dimensions. Width and height must be positive and within "
                << original_width << " x " << original_height << "." << endl;
            continue; // Prompt again if dimensions are out of bounds
        }

        // Set the validated dimensions
        new_width = width;
        new_height = height;

        int num_vertical_seams = original_width - new_width;
        int num_horizontal_seams = original_height - new_height;

        if (num_vertical_seams < 0 || num_horizontal_seams < 0) {
            cout << "New dimensions must be smaller than or equal to the original dimensions." << endl;
            continue;
        }

        Mat image_dp = original_image.clone();
        Mat image_greedy = original_image.clone();

        for (int i = 0; i < num_vertical_seams; i++) {
            removeVerticalSeamDP(image_dp);
        }
        for (int i = 0; i < num_horizontal_seams; i++) {
            removeHorizontalSeamDP(image_dp);
        }

        for (int i = 0; i < num_vertical_seams; i++) {
            removeVerticalSeamGreedy(image_greedy);
        }
        for (int i = 0; i < num_horizontal_seams; i++) {
            removeHorizontalSeamGreedy(image_greedy);
        }

        stringstream ss_filename_dp, ss_filename_greedy;
        ss_filename_dp << "output_dp_" << new_width << "x" << new_height << ".png";
        ss_filename_greedy << "output_greedy_" << new_width << "x" << new_height << ".png";

        // Save the results with fixed filenames
        imwrite("output_dp.png", image_dp);
        imwrite("output_greedy.png", image_greedy);

        namedWindow("Original Image", WINDOW_AUTOSIZE);
        imshow("Original Image", original_image);

        namedWindow("Dynamic Programming Result", WINDOW_AUTOSIZE);
        imshow("Dynamic Programming Result", image_dp);

        namedWindow("Greedy Algorithm Result", WINDOW_AUTOSIZE);
        imshow("Greedy Algorithm Result", image_greedy);

        waitKey(0);
        destroyAllWindows(); // Close the image windows before the next iteration
    }

    return 0;
}

