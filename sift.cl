__kernel void conv_and_trans(
	__global float *dst, __global float *src,
	__global const float *kern, int kernSiz,
	int w, int h
)
{
	int X = get_global_id(0);
	if (X >= w) {
		return ;
	}
	__global const float *row = src;
	__global float *col = dst + X*h;
	for	(int i = 0; i < h; ++i) {
		float sum = 0.0f;
		for (int j = -kernSiz; j <= kernSiz; ++j) {
			int x = X+j;
			x = max(x, 0);
			x = min(x, w-1);
			sum += row[x] * kern[j+kernSiz];
			// sum += row[x];
		}
		*col = sum;
		// *col = sum / (1+2*kernSiz);
		++col;
		row += w;
	}

	/*
	__global const float *row = src+X;
	__global float *col = dst + X*h;
	for	(int i = 0; i < h; ++i) {
		*col = *row;
		++col;
		row += w;
	}
	*/
}

__kernel void diff(
	__global float *dog, __global float *gauss,
	int s, int w, int h
)
{
	int X = get_global_id(0);
	if (X >= w) {
		return ;
	}
	__global float *src1 = gauss + X;
	__global float *src2 = gauss + X + w*h;
	__global float *dst = dog + X;
	for	(int i = 0; i < (s-1)*h; ++i) {
		*dst = *src2 - *src1;

		src1 += w;
		src2 += w;
		dst += w;
	}
}
