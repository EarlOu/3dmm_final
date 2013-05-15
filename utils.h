#ifndef __UTILS_H__
#define __UTILS_H__

#define MAX(x, y) ((x)>(y)?(x):(y))
#define MIN(x, y) ((x)<(y)?(x):(y))

void diff(float *octave3d, int s, int w, int h);
void upSample2(float *dst, float *src, float *buf, int w, int h);
void downSample(float *dst, float *src, int w, int h, int d);
void conv1D_symm_and_transpose(
	float *out, float *in, int w, int h,
	int kernelSize, float *kernel);
void gaussian_blur(float *out, float *in, float *buf, int w, int h, float sigma);
void matrix_multiply(float* y, float* A, float* x);
void rotate_point(float& dstX, float& dstY, float srcX, float srcY,float theta);
void build_hessian(float* hessian, float* point, int w, int h);
void build_gradient(float* gradient, float* point, int w, int h);
void inv_3d_matrix(float* dst, float* src);

#endif
