
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t difuseImage(std::string image_name);


std::string key;
cv::Mat image;
std::vector<long int> M, N, A, B, D, E;
std::vector< std::vector<long int> > x;
int m, n;
double mu;
double epsilon;

long int f(long int x) {
    long int val = 1 - x;
    long int res = (mu * x * val);
    return abs(res % 256);
}


void CML(int numberOfTimes) {
    long double mu_sum = 0;
    std::string mu_key = key.substr(0, 10);
    for (size_t i = 0; i < 10; i++)
    {
        mu_sum += (int)mu_key[i] * pow(2, i);
    }

    mu_sum /= pow(2, 10);
    mu = 3.99 + 0.01 * mu_sum;

    long double epsilon_sum = 0;
    std::string epsilon_key = key.substr(10, 10);
    for (size_t i = 0; i < 10; i++)
    {
        epsilon_sum += (int)epsilon_key[i] * pow(2, i);
    }

    epsilon_sum /= pow(2, 10);
    epsilon = 0.1 * epsilon_sum;
    std::string temp_key;
    long int temp_key_value = 0;
    std::vector<long int> x_0;
    for (int i = 20; i < key.length(); i += 10)
    {
        temp_key = key.substr(i, 10);
        for (size_t i = 0; i < 10; i++)
        {
            int t = (int)temp_key[i];
            int exp = (int)pow(2, i);
            temp_key_value = temp_key_value + t * exp;


        }

        temp_key_value /= pow(2, 10);
        temp_key_value = temp_key_value % 256;
        x_0.push_back(temp_key_value);
        temp_key_value = 0;
    }
    x.push_back(x_0);
    std::vector<long int> temp_x;
    for (size_t i = 1; i <= numberOfTimes; i++)
    {
        for (size_t j = 0; j < (x[i - 1]).size(); j++)
        {
            temp_x.push_back(f(x[i - 1][j]));
        }

        x.push_back(temp_x);
        temp_x.clear();
    }
}

void generateKeys(int p) {
    int val = (2 * p + 4);
    for (size_t i = 0; i < x.size(); i++)
    {

        if (i >= (x.size() - m))
        {
            M.push_back(x[i][0]);
        }

        if (i >= (x.size() - n))
        {
            N.push_back(x[i][1]);
        }

        if (i >= (x.size() - val))
        {
            A.push_back(x[i][2]);
            B.push_back(x[i][3]);
            D.push_back(x[i][4]);
            E.push_back(x[i][5]);
        }


    }
}

void shiftDown(uint8_t* arr, size_t l) {
    uint8_t temp = arr[l - 1];
    for (size_t i = 0; i < l - 1; i++)
    {
        arr[i] = arr[i + 1];
    }

    arr[0] = temp;
}

uint8_t* permutateImage(std::vector<long int> row_key, std::vector<long int> reference_row_key, std::vector<long int> col_key, std::vector<long int> reference_col_key, uint8_t* image)
{
    uint8_t* res = new uint8_t[m * n];
    for (size_t i = 0; i < reference_row_key.size(); i++)
    {
        for (size_t j = 0; j < row_key.size(); j++)
        {
            if (row_key[j] == reference_row_key[i])
            {
                for (size_t k = 0; k < n; k++)
                {
                    res[k * m + j] = image[k * m + j];
                }
                break;
            }
        }
    }

    for (size_t i = 0; i < reference_col_key.size(); i++)
    {
        for (size_t j = 0; j < col_key.size(); j++)
        {
            if (col_key[j] == reference_col_key[i])
            {
                for (size_t k = 0; k < m; k++)
                {
                    res[j * m + k] = image[j * m + k];
                }
            }
        }
    }

    return res;


}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void diffuseImageKernel(unsigned char* imageOut, unsigned char* imageIn,long long int* G, size_t size_G,long long int* X, size_t size_X, unsigned int h, unsigned int w) {
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    if ((row < h) && (col < w))
    {
        unsigned index = row * w + col;
        imageOut[index] = (G[index] + X[index] * 100000 + imageIn[index]) % 256;
    }
    
}


int main()
{
    key = "8123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567";
    image = cv::imread("lena1024.jpg");
    m = image.rows;
    n = image.cols;
    time_t start, end;
    int p = 4;
    //cout<<(int) image.at<uchar>(0,0)<<endl;
    int temp_max = std::max(m, n);
    int cmlTimes = 200 + std::max(temp_max, 2 * p + 4);
    CML(cmlTimes);
    /*cout << mu << " " << epsilon << endl;
    for (size_t i = 0; i < x.size(); i++)
    {
        for (size_t j = 0; j < (x[i]).size(); j++)
        {
            cout << x[i][j] << " ";
        }
        cout << endl;
    }*/

    generateKeys(p);
    size_t image_size = image.cols * image.rows;
    uint8_t* data_image = new uint8_t[image_size];
    memcpy(data_image, image.data, image_size);
    std::vector<long int> sort_M, sort_N;
    sort_M = M;
    sort_N = N;
    std::sort(sort_M.begin(), sort_M.end());
    std::sort(sort_N.begin(), sort_N.end());

    auto t1 = std::chrono::high_resolution_clock::now();
    uint8_t* res = permutateImage(M, sort_M, N, sort_N, data_image);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Permutation Time : " << duration;
    std::cout << " ms " << std::endl;
    cv::Mat image_res(m, n, CV_8UC1, res);
    cv::imwrite("res.png", image_res);
    
    cudaError_t cudaStatus = difuseImage("res.png");
   
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}



cudaError_t difuseImage(std::string image_name)
{
    cv::Mat image = cv::imread(image_name);
    unsigned int h = image.rows;
    unsigned int w = image.cols;
    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(ceil(w / 16), ceil(h / 16), 1);
    size_t size_image = sizeof(unsigned char) * h * w;
    unsigned char* h_image_data = image.data;

    std::vector<long long int> G;
    std::vector<long long int> X;
    G.push_back((A[0] * 100000) % 256);
    G.push_back((B[1] * 100000) % 256);
    X.push_back(E[0]);
    for (size_t i = 2; i < size_image; i++)
    {
        long int temp = floor(G[i - 2] + 100000 * G[i - 1] / 255);
        G.push_back(temp);
    }

    for (size_t i = 1; i < size_image; i++)
    {
        float mu = 3.99 + 0.01 * D[i % D.size()];
        int temp_x = mu * X[i - 1] * (1 - X[i - 1]);
        X.push_back(temp_x);
    }

    
    long long int *h_G = G.data();
    long long int *h_X = X.data();
    long long int *d_G,*d_X;
    unsigned char* d_image_data;
    unsigned char* d_image_result;
    
    float elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMalloc((void**)&d_image_data,size_image);
    cudaMalloc((void**)&d_image_result, size_image);
    cudaMalloc((void**)&d_G, G.size());
    cudaMalloc((void**)&d_X, X.size());

    cudaMemcpy(d_image_data, h_image_data,size_image, cudaMemcpyHostToDevice);
    cudaMemcpy(d_G, h_G, G.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, X.size(), cudaMemcpyHostToDevice);

    diffuseImageKernel <<< dimGrid, dimBlock >>> (d_image_result, d_image_data, d_G, G.size(), d_X, X.size(), h, w);
    cudaMemcpy(h_image_data, d_image_result, size_image, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("The elapsed time in gpu diffused %.4f ms\n", elapsed );

    cv::Mat result(h, w, CV_8UC1, h_image_data);

    cv::imwrite("encripted.png", result);

    return cudaError_t();
}
