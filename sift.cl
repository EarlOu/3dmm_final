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
