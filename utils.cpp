#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "utils.h"

void diff(float *dog, float *blurred, int s, int w, int h);
{
	for (int i = 0; i < (s-1)*w*h; ++i) {
		octave3d[0] = blurred[w*h] - blurred[0];
		++blurred;
		++octave3d;
	}
}

void build_gradient_map(float *map, float *blurred, int _s, int w, int h)
{
	for (int s = 0; s < _s; ++s) {
		map += w*2;
		blurred += w;
		for (int i = 1; i < (h - 1); ++i) {
			map += 2;
			++blurred;
			for (int j = 1; j < (w - 1); ++j) {
				float dx = 0.5 * (blurred[1] - blurred[-1]);
				float dy = 0.5 * (blurred[w] - blurred[-w]);
				++blurred;
				theta = atan2f(dy, dx);
				if (theta < 0) {
					theta += 2 * M_PI;
				}

				++map;
				*map = sqrtf(dx * dx + dy * dy);
				++map;
				*map = theta;
			}
			map += 2;
			++blurred;
		}
		map += w*2;
		blurred += w;
	}
}

static void upSampleRowAndTranspose(float *dst, float *src, int w, int h)
{
	int dstRewind = (w*2 - 1) * h - 1;
	for (int i = 0; i < h; ++i) {
		float a, b;
		b = *src;
		for (int j = 0; j < w-1; ++j) {
			a = b;
			b = *src;
			++src;

			*dst = a;
			dst += h;
			*dst = 0.5f * (a+b);
			dst += h;
		}

		// copy the last pixel twice
		*dst = b;
		dst += h;
		*dst = b;

		dst -= dstRewind;
		++src;
	}
}

void upSample2(float *dst, float *src, float *buf, int w, int h)
{
	upSampleRowAndTranspose(buf, src, w, h);
	upSampleRowAndTranspose(dst, buf, h, w<<1);
}

void downSample(float *dst, float *src, int w, int h, int d)
{
	for (int i = 0; i < h-d+1; i += d) {
		float *rowstart = src + i*w;
		for (int j = 0; j < w-d+1; j += d) {
			*dst = *rowstart;
			++dst;
			rowstart += d;
		}
	}
}

static int generate_1D_gaussian_kernel(float* &kernel, float sigma)
{
	int kernelSize = ceil(4 * sigma);
	kernel = new float[2*kernelSize + 1];
	// create right side
	for (int i = 0; i < (kernelSize + 1); ++i) {
		kernel[i + kernelSize] = 1 / (sigma * sqrtf(2 * M_PI)) *
			exp(-1 * pow(i, 2) / (2 * pow(sigma, 2)));
	}
	// reverse right side as left side
	for (int i = 0; i < kernelSize; ++i) {
		kernel[i] = kernel[2 * kernelSize - i];
	}
	// normalize
	float sum = 0;
	for (int i = 0; i < (2 * kernelSize + 1); ++i) {
		sum += kernel[i];
	}
	float isum = 1.0f / sum;
	for (int i = 0; i < (2 * kernelSize + 1); ++i) {
		kernel[i] *= isum;
	}

	return kernelSize;
}

void gaussian_blur(float *out, float *in, float *buf, int w, int h,
	float sigma)
{
	float* kernel;
	int kernelSize = generate_1D_gaussian_kernel(kernel, sigma);

	// cal the kernel
	conv1D_symm_and_transpose(buf,  in, w, h, kernelSize, kernel);
	conv1D_symm_and_transpose(out, buf, h, w, kernelSize, kernel);

	delete[] kernel;

	return;
}

void conv1D_symm_and_transpose(float *out, float *in, int w, int h,
	int kernelSize, float *kernel)
{
	float sum;
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			sum = 0;
			for (int k = (j-kernelSize); k < 0; ++k) {
				sum += in[i*w] * kernel[k-j+kernelSize];
			}
			for (int k = MAX(0, j-kernelSize); k < MIN(w, j+kernelSize+1); ++k) {
				sum += in[i*w+k] * kernel[k-j+kernelSize];
			}
			for (int k = w; k < (j+kernelSize+1); ++k) {
				sum += in[i*w+w-1] * kernel[k-j+kernelSize];
			}
			out[j*h+i] = sum;
		}
	}
	return;
}

void inv_3d_matrix(float* dst, float* src)
{
#define ARR(P, X, Y) (*(P + (3 * (X-1)) + (Y-1))) // (X, Y) = (1, 1) ~ (3, 3)
	// determinant
	float det = + ARR(src,1,1) * ARR(src,2,2) * ARR(src,3,3)
		+ ARR(src,1,2) * ARR(src,2,3) * ARR(src,3,1)
		+ ARR(src,1,3) * ARR(src,2,1) * ARR(src,3,2)
		- ARR(src,1,1) * ARR(src,2,3) * ARR(src,3,2)
		- ARR(src,1,2) * ARR(src,2,1) * ARR(src,3,3)
		- ARR(src,1,3) * ARR(src,2,2) * ARR(src,3,1);
	dst[0] = ((ARR(src,2,2)*ARR(src,3,3)) - (ARR(src,3,2)*ARR(src,2,3))) / det;
	dst[1] = ((ARR(src,3,2)*ARR(src,1,3)) - (ARR(src,1,2)*ARR(src,3,3))) / det;
	dst[2] = ((ARR(src,1,2)*ARR(src,2,3)) - (ARR(src,2,2)*ARR(src,1,3))) / det;
	dst[3] = ((ARR(src,2,3)*ARR(src,3,1)) - (ARR(src,3,3)*ARR(src,2,1))) / det;
	dst[4] = ((ARR(src,3,3)*ARR(src,1,1)) - (ARR(src,1,3)*ARR(src,3,1))) / det;
	dst[5] = ((ARR(src,1,3)*ARR(src,2,1)) - (ARR(src,2,3)*ARR(src,1,1))) / det;
	dst[6] = ((ARR(src,2,1)*ARR(src,3,2)) - (ARR(src,3,1)*ARR(src,2,2))) / det;
	dst[7] = ((ARR(src,3,1)*ARR(src,1,2)) - (ARR(src,1,1)*ARR(src,3,2))) / det;
	dst[8] = ((ARR(src,1,1)*ARR(src,2,2)) - (ARR(src,2,1)*ARR(src,1,2))) / det;
	return;
#undef ARR
}

void matrix_multiply(float* y, float* A, float* x)
{
	for (int i = 0; i < 3; ++i) {
		y[i] = 0;
		for (int j = 0; j < 3; ++j) {
			y[i] += A[i * 3 + j] * x[j];
		}
	}
	return;
}

void rotate_point(float& dstX, float& dstY, float srcX, float srcY, float theta)
{
	float sinTheta = sinf(theta / 180 * M_PI);
	float cosTheta = cosf(theta / 180 * M_PI);
	dstX = cosTheta * srcX - sinTheta * srcY;
	dstY = sinTheta * srcX + cosTheta * srcY;
	return;
}

#define VAL(P, X, Y, Z, S_X, S_Y, S_Z) (*(P + (X*S_X) + (Y*S_Y) + (Z*S_Z)))
void build_hessian(float* hessian, float* point, int w, int h)
{
  float Dxx, Dyy, Dss, Dxy, Dxs, Dys;
  Dxx =  + VAL(point,  1,  0,  0, 1, w, (w * h))
         - VAL(point,  0,  0,  0, 1, w, (w * h)) * 2.0f
         + VAL(point, -1,  0,  0, 1, w, (w * h));
  Dxy = (+ VAL(point,  1,  1,  0, 1, w, (w * h))
         - VAL(point,  1, -1,  0, 1, w, (w * h))
         - VAL(point, -1,  1,  0, 1, w, (w * h))
         + VAL(point, -1, -1,  0, 1, w, (w * h))) / 4.0f;
  Dxs = (+ VAL(point,  1,  0,  1, 1, w, (w * h))
         - VAL(point,  1,  0, -1, 1, w, (w * h))
         - VAL(point, -1,  0,  1, 1, w, (w * h))
         + VAL(point, -1,  0, -1, 1, w, (w * h))) / 4.0f;
  Dyy =  + VAL(point,  0,  1,  0, 1, w, (w * h))
         - VAL(point,  0,  0,  0, 1, w, (w * h)) * 2.0f
         + VAL(point,  0, -1,  0, 1, w, (w * h));
  Dys = (+ VAL(point,  0,  1,  1, 1, w, (w * h))
         - VAL(point,  0,  1, -1, 1, w, (w * h))
         - VAL(point,  0, -1,  1, 1, w, (w * h))
         + VAL(point,  0, -1, -1, 1, w, (w * h))) / 4.0f;
  Dss =  + VAL(point,  0,  0,  1, 1, w, (w * h))
         - VAL(point,  0,  0,  0, 1, w, (w * h)) * 2.0f
         + VAL(point,  0,  0, -1, 1, w, (w * h));
  hessian[0] = Dxx;
  hessian[1] = hessian[3] = Dxy;
  hessian[2] = hessian[6] = Dxs;
  hessian[4] = Dyy;
  hessian[5] = hessian[7] = Dys;
  hessian[8] = Dss;
}

void build_gradient(float* gradient, float* point, int w, int h)
{
  gradient[0] = (+ VAL(point,  1,  0,  0, 1, w, (w * h))
                 - VAL(point, -1,  0,  0, 1, w, (w * h))) / 2.0f;
  gradient[1] = (+ VAL(point,  0,  1,  0, 1, w, (w * h))
                 - VAL(point,  0, -1,  0, 1, w, (w * h))) / 2.0f;
  gradient[2] = (+ VAL(point,  0,  0,  1, 1, w, (w * h))
                 - VAL(point,  0,  0, -1, 1, w, (w * h))) / 2.0f;
}
#undef VAL
