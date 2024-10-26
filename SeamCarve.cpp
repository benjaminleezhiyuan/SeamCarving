#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Function to compute the energy map of the image
Mat computeEnergyMap(const Mat& image) {
    Mat gray, grad_x, grad_y, abs_grad_x, abs_grad_y, energy_map;

    // Convert the input image to grayscale
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Compute gradients along the X-axis using the Sobel operator
    Sobel(gray, grad_x, CV_16S, 1, 0, 3);

    // Compute gradients along the Y-axis using the Sobel operator
    Sobel(gray, grad_y, CV_16S, 0, 1, 3);

    // Convert the gradient images to absolute values
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    // Combine the absolute gradients to form the energy map
    // Each pixel's energy is the sum of its absolute gradients in X and Y
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, energy_map);

    return energy_map;
}

// Function to find and remove a vertical seam using dynamic programming
void removeVerticalSeamDP(Mat& image) {
    // Compute the energy map of the current image
    Mat energy_map = computeEnergyMap(image);
    int rows = energy_map.rows;
    int cols = energy_map.cols;

    // Initialize the cumulative energy map with zeros
    Mat M = Mat::zeros(rows, cols, CV_32S);

    // Copy the first row of the energy map to the cumulative energy map
    energy_map.row(0).convertTo(M.row(0), CV_32S);

    // Compute the cumulative energy map by dynamic programming
    for (int i = 1; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Start with the energy from the pixel directly above
            int min_energy = M.at<int>(i - 1, j);

            // Check the pixel to the top-left, if it exists
            if (j > 0)
                min_energy = min(min_energy, M.at<int>(i - 1, j - 1));

            // Check the pixel to the top-right, if it exists
            if (j < cols - 1)
                min_energy = min(min_energy, M.at<int>(i - 1, j + 1));

            // Update the cumulative energy for the current pixel
            M.at<int>(i, j) = energy_map.at<uchar>(i, j) + min_energy;
        }
    }

    // Backtrack to find the path of the seam with the minimum energy
    vector<int> seam(rows);
    int min_idx = 0;
    int min_val = M.at<int>(rows - 1, 0);

    // Find the position in the last row with the minimum cumulative energy
    for (int j = 1; j < cols; j++) {
        if (M.at<int>(rows - 1, j) < min_val) {
            min_val = M.at<int>(rows - 1, j);
            min_idx = j;
        }
    }
    seam[rows - 1] = min_idx;

    // Trace the seam path from bottom to top
    for (int i = rows - 2; i >= 0; i--) {
        int prev_x = seam[i + 1];
        int min_energy = M.at<int>(i, prev_x);
        min_idx = prev_x;

        // Check the top-left neighbor
        if (prev_x > 0 && M.at<int>(i, prev_x - 1) < min_energy) {
            min_energy = M.at<int>(i, prev_x - 1);
            min_idx = prev_x - 1;
        }

        // Check the top-right neighbor
        if (prev_x < cols - 1 && M.at<int>(i, prev_x + 1) < min_energy) {
            min_energy = M.at<int>(i, prev_x + 1);
            min_idx = prev_x + 1;
        }

        // Update the seam path
        seam[i] = min_idx;
    }

    // Create an output image with one less column to remove the seam
    Mat output(rows, cols - 1, CV_8UC3);

    // Iterate through each row to remove the seam
    for (int i = 0; i < rows; i++) {
        int idx = seam[i];
        for (int j = 0; j < idx; j++) {
            // Copy pixels before the seam
            output.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
        }
        for (int j = idx + 1; j < cols; j++) {
            // Shift pixels after the seam to the left by one
            output.at<Vec3b>(i, j - 1) = image.at<Vec3b>(i, j);
        }
    }

    // Update the original image with the seam removed
    image = output.clone();
}

// Function to find and remove a horizontal seam using dynamic programming
void removeHorizontalSeamDP(Mat& image) {
    // Transpose the image to reuse the vertical seam removal function
    Mat transposed_image;
    transpose(image, transposed_image);

    // Flip the transposed image vertically to maintain orientation
    flip(transposed_image, transposed_image, 0);

    // Remove a vertical seam from the transposed image
    removeVerticalSeamDP(transposed_image);

    // Flip the image back vertically after seam removal
    flip(transposed_image, transposed_image, 0);

    // Transpose the image back to its original orientation
    transpose(transposed_image, image);
}

// Function to find and remove a vertical seam using a greedy algorithm
void removeVerticalSeamGreedy(Mat& image) {
    // Compute the energy map of the current image
    Mat energy_map = computeEnergyMap(image);
    int rows = energy_map.rows;
    int cols = energy_map.cols;

    // Initialize the seam path
    vector<int> seam(rows);

    // Start from the top row and find the pixel with the minimum energy
    double min_val;
    Point min_loc;
    minMaxLoc(energy_map.row(0), &min_val, nullptr, &min_loc, nullptr);
    seam[0] = min_loc.x;

    // Greedily find the seam path by selecting the minimum energy neighbor at each step
    for (int i = 1; i < rows; i++) {
        int prev_x = seam[i - 1];
        int min_energy = energy_map.at<uchar>(i, prev_x);
        int min_idx = prev_x;

        // Check the left neighbor
        if (prev_x > 0 && energy_map.at<uchar>(i, prev_x - 1) < min_energy) {
            min_energy = energy_map.at<uchar>(i, prev_x - 1);
            min_idx = prev_x - 1;
        }

        // Check the right neighbor
        if (prev_x < cols - 1 && energy_map.at<uchar>(i, prev_x + 1) < min_energy) {
            min_energy = energy_map.at<uchar>(i, prev_x + 1);
            min_idx = prev_x + 1;
        }

        // Update the seam path
        seam[i] = min_idx;
    }

    // Create an output image with one less column to remove the seam
    Mat output(rows, cols - 1, CV_8UC3);

    // Iterate through each row to remove the seam
    for (int i = 0; i < rows; i++) {
        int idx = seam[i];
        for (int j = 0; j < idx; j++) {
            // Copy pixels before the seam
            output.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
        }
        for (int j = idx + 1; j < cols; j++) {
            // Shift pixels after the seam to the left by one
            output.at<Vec3b>(i, j - 1) = image.at<Vec3b>(i, j);
        }
    }

    // Update the original image with the seam removed
    image = output.clone();
}

// Function to find and remove a horizontal seam using a greedy algorithm
void removeHorizontalSeamGreedy(Mat& image) {
    // Transpose the image to reuse the vertical seam removal function
    Mat transposed_image;
    transpose(image, transposed_image);

    // Flip the transposed image vertically to maintain orientation
    flip(transposed_image, transposed_image, 0);

    // Remove a vertical seam from the transposed image using the greedy approach
    removeVerticalSeamGreedy(transposed_image);

    // Flip the image back vertically after seam removal
    flip(transposed_image, transposed_image, 0);

    // Transpose the image back to its original orientation
    transpose(transposed_image, image);
}

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

        // Load the image from the specified file
        original_image = imread(filename);

        // Check if the image was loaded successfully
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

    // Main loop to process user inputs for resizing
    while (true) {
        int new_width = -1, new_height = -1;
        cout << "Enter the desired new width and height (e.g., 500 500), 'new' to load a new image, or '-1' to exit: ";
        string input;
        getline(cin, input);

        // Check if the user wants to exit the program
        if (input == "-1") {
            break; // Exit the loop and terminate the program
        }
        else if (input == "new") {
            // Inner loop to load a new image if the user enters "new"
            while (true) {
                cout << "Enter the new image name (without extension): ";
                cin >> filename;
                cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Clear the newline character

                // Append .png to the filename
                filename += ".png";

                // Load the new image from the specified file
                original_image = imread(filename);

                // Check if the image was loaded successfully
                if (!original_image.empty()) {
                    break; // Exit the loop if a valid image is loaded
                }

                // If loading fails, display an error and prompt again
                cout << "Could not open or find the image! Please try again." << endl;
            }

            // Update the original image dimensions after loading a new image
            original_width = original_image.cols;
            original_height = original_image.rows;
            cout << "Original image dimensions: " << original_width << " x " << original_height << endl;
            continue; // Go back to the beginning of the loop for new input
        }

        // Use a stringstream to parse the input for width and height
        stringstream ss(input);
        int width, height;
        if (!(ss >> width >> height) || (ss >> ws, !ss.eof())) {
            // If parsing fails or there are extra characters, prompt again
            cout << "Invalid input. Please enter exactly two integer values for width and height." << endl;
            continue; // Prompt again if input is invalid
        }

        // Check that the new dimensions are within valid bounds
        if (width <= 0 || width > original_width || height <= 0 || height > original_height) {
            cout << "Invalid dimensions. Width and height must be positive and within "
                << original_width << " x " << original_height << "." << endl;
            continue; // Prompt again if dimensions are out of bounds
        }

        // Set the validated new dimensions
        new_width = width;
        new_height = height;

        // Calculate the number of seams to remove for width and height
        int num_vertical_seams = original_width - new_width;
        int num_horizontal_seams = original_height - new_height;

        // Ensure that the new dimensions are smaller than or equal to the original
        if (num_vertical_seams < 0 || num_horizontal_seams < 0) {
            cout << "New dimensions must be smaller than or equal to the original dimensions." << endl;
            continue;
        }

        // Clone the original image to create separate images for DP and Greedy methods
        Mat image_dp = original_image.clone();
        Mat image_greedy = original_image.clone();

        // Remove vertical seams using Dynamic Programming
        for (int i = 0; i < num_vertical_seams; i++) {
            removeVerticalSeamDP(image_dp);
        }

        // Remove horizontal seams using Dynamic Programming
        for (int i = 0; i < num_horizontal_seams; i++) {
            removeHorizontalSeamDP(image_dp);
        }

        // Remove vertical seams using the Greedy algorithm
        for (int i = 0; i < num_vertical_seams; i++) {
            removeVerticalSeamGreedy(image_greedy);
        }

        // Remove horizontal seams using the Greedy algorithm
        for (int i = 0; i < num_horizontal_seams; i++) {
            removeHorizontalSeamGreedy(image_greedy);
        }

        // Prepare filenames for saving the output images
        stringstream ss_filename_dp, ss_filename_greedy;
        ss_filename_dp << "output_dp_" << new_width << "x" << new_height << ".png";
        ss_filename_greedy << "output_greedy_" << new_width << "x" << new_height << ".png";

        // Save the results with fixed filenames
        imwrite("output_dp.png", image_dp);
        imwrite("output_greedy.png", image_greedy);

        // Display the original and processed images in separate windows
        namedWindow("Original Image", WINDOW_AUTOSIZE);
        imshow("Original Image", original_image);

        namedWindow("Dynamic Programming Result", WINDOW_AUTOSIZE);
        imshow("Dynamic Programming Result", image_dp);

        namedWindow("Greedy Algorithm Result", WINDOW_AUTOSIZE);
        imshow("Greedy Algorithm Result", image_greedy);

        // Wait for a key press to proceed
        waitKey(0);

        // Close all image windows before the next iteration
        destroyAllWindows();
    }

    return 0; // Exit the program
}
