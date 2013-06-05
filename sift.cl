#include "clshare.h"

struct Keypoint {
	int ix, iy, is, o;
	float x, y, sigma, orient;
};

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

__kernel void conv_and_trans2(
	__global float *dst, __global float *src,
	__global const float *kern, int kernSiz,
	int w, int h
)
{
	int X = get_global_id(0);
	if (X >= w) {
		return ;
	}

	float buffer[2*MAX_KERNSIZ+1];
	__global float *src_tmp = src + X;
	buffer[0] = *src_tmp;
	for (int i = 0; i <= kernSiz; ++i) {
		buffer[i] = buffer[0];
	}
	for (int i = 1; i <= kernSiz; ++i) {
		src_tmp += w;
		buffer[kernSiz+i] = *src;
	}

	int start = 0;
	__global float *dst_tmp = dst + h*X;
	for	(int i = 0; i < h; ++i) {
		float sum = 0.0f;
		for (int j = 0; j <= 2*kernSiz; ++j) {
			int ind = j+start;
			if (ind > 2*kernSiz) {
				ind -= 2*kernSiz+1;
			}
			sum += kern[j] * buffer[ind];
		}
		*dst_tmp = sum;
		++dst_tmp;
		if (i < h-kernSiz) {
			src_tmp += w;
		}

		++start;
		if (start > 2*kernSiz) {
			start = 0;
		}
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

inline __global float * get_gradient(int o, __global float* magAndThetas,
	int wmax, int hmax, int lvPerScale)
{
	int offset = 0, wtmp = wmax, htmp = hmax;
	int i;
	for (i = 0; i < o; ++i) {
		offset += 2 * wtmp * htmp * lvPerScale;
		wtmp >>= 1;
		htmp >>= 1;

	}
	return magAndThetas + offset;
}

inline void normalize_feature(__global float* v)
{
	float sqsum = 0.0f;
	for (int i = 0; i < 128; ++i) {
		sqsum += v[i] * v[i];
	}
	float inv = 1.0f/sqrt(sqsum);
	for (int i = 0; i < 128; ++i) {
		v[i] *= inv;
	}
}

inline void regularize_feature(__global float* v)
{

	normalize_feature(v);
	for (int i = 0; i < 128; ++i) {
		if (v[i] > 0.2f) {
			v[i] = 0.2f;
		}
	}
	normalize_feature(v);
}

__kernel void calc_kp_descriptors(
	__global float *dess,
	__global const struct Keypoint *kps,
	__global float * magAndThetas,
	int wmax, int hmax, int lvPerScale, int n)
{
	int X = get_global_id(0);
	if (X >= n) return;
	const struct Keypoint kp = kps[X];

	int o = kp.o;
	int s = kp.is;
	float period = pown(2.0f, o);
	int w = wmax >> o;
	int h = hmax >> o;

	__global float* gradImage = get_gradient(o, magAndThetas, wmax, hmax, lvPerScale) + s*w*h*2;
	float sigma = kp.sigma / period;
	float floatX = kp.x / period;
	float floatY = kp.y / period;
	int intX = (int)(floatX + 0.5);
	int intY = (int)(floatY + 0.5);

	if (intX < 1 || intX >= h || intY < 1 || intY >= w) {
		return;
	}
	
	// 8 orientations x 4 x 4 histogram array = 128 dimensions
	const int NBO = 8; // number of orientations
	const int NBP = 4; // number of small blocks
	const float SBP = (3.0f * sigma); // size of a small block
	const int W = (int)floor((SBP * (NBP+1) * sqrt(0.5)) + 0.5); // W=SBP(NBP+1)/sqrt(2)

	float angle0 = kp.orient;

	const float st0 = sin(angle0);
	const float ct0 = cos(angle0);
	
	const int binto = 1;
	const int binyo = NBO * NBP;
	const int binxo = NBO;

	/* Center the scale space and the descriptor on the current keypoint.
	 * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
	 */
	__global float const * pt = gradImage + (intX + intY * w) * 2;
	__global float*       dpt = dess + (128 * X) + (NBP/2) * binyo + (NBP/2) * binxo ;

#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)
#define MAX(a, b) ((a>b)?a:b)
#define MIN(a, b) ((a<b)?a:b)
#define ABS(a) ((a>0)?a:-a)

	for (int i = MAX(-W, 1-intY); i < MIN(W+1, h-1-intY); ++i) {
		for (int j = MAX(-W, 1-intX); j < MIN(W+1, w-1-intX); ++j) {
			// start copying siftpp() ... so 'dy' and 'dx' conform to its convention
			float dx = j + intX - floatX;
			float dy = i + intY - floatY;

			// for the sample point
			float mod = gradImage[((intY + i) * w + (intX + j)) * 2];
			float angle = gradImage[((intY + i)*w + (intX + j)) * 2+1];
			float theta = (-angle + angle0);
			if (theta < 0) {
				theta += 2 * M_PI;
			}

			// get the displacement normalized w.r.t. the keypoint
			// orientation and extension.
			float nx = ( ct0 * dx + st0 * dy) / SBP ;
			float ny = (-st0 * dx + ct0 * dy) / SBP ;
			float nt = NBO * theta / (2*M_PI) ;

			// Get the gaussian weight of the sample. The gaussian window
			// has a standard deviation equal to NBP/2. Note that dx and dy
			// are in the normalized frame, so that -NBP/2 <= dx <= NBP/2.
			float const wsigma = NBP/2 ;
			float win = exp(-(nx*nx + ny*ny)/(2.0 * wsigma * wsigma)) ;

			// The sample will be distributed in 8 adjacent bins.
			// We start from the ``lower-left'' bin.
			int binx = floor( nx - 0.5 ) ;
			int biny = floor( ny - 0.5 ) ;
			int bint = floor( nt ) ;

			//rbinx net result ==> (nx-0.5)-floor(nx-0.5)
			//rbiny net result ==> (ny-0.5)-floor(ny-0.5)
			float rbinx = nx - (binx+0.5) ;
			float rbiny = ny - (biny+0.5) ;
			float rbint = nt - bint ;
			int dbinx ;
			int dbiny ;
			int dbint ;

			// Distribute the current sample into the 8 adjacent bins
			for(dbinx = 0 ; dbinx < 2 ; ++dbinx) {
				for(dbiny = 0 ; dbiny < 2 ; ++dbiny) {
					for(dbint = 0 ; dbint < 2 ; ++dbint) {

						if( binx+dbinx >= -(NBP/2) &&
							binx+dbinx <   (NBP/2) &&
							biny+dbiny >= -(NBP/2) &&
							biny+dbiny <   (NBP/2) ) {
							float weight = win
								* mod
								* ABS (1 - dbinx - rbinx)
								* ABS (1 - dbiny - rbiny)
								* ABS (1 - dbint - rbint) ;

							atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight ;
						}
					}
				}
			}
		}
	}
	regularize_feature(dess + 128 * X);
}

__kernel void calc_angle(
				__global struct Keypoint *kp,
				__global void *magAndThetas,
				int wmax, int hmax, int lvPerScale, int n)
{
	
	int index = get_global_id(0);
	if (index >= n) return;
	int o = kp[index].o;
	int s = kp[index].is;
	float period = 1;
	for (int i=0;i<o;i++)
	{
		period *=2.0;
	}
	float sigmaW = 1.5f * (kp[index].sigma / period);
	int windowSize = 3 * sigmaW;
	if (windowSize <= 0) {
		windowSize = 1;
	}
	int w = wmax>>o;
	int h = hmax>>o;
	
	__global float* gradImage = get_gradient(o, magAndThetas, wmax, hmax, lvPerScale) + s*w*h*2;
    
	float floatX = kp[index].x / period;
	float floatY = kp[index].y / period;
	int intX = (int)(floatX + 0.5);
	int intY = (int)(floatY + 0.5);
    
	if (intX < 1 || intX >= w || intY < 1 || intY >= h) {
		return;
	}
    
	// compute histogram
	const int histSize = 36;
	float hist[36];
	for (int i=0;i<histSize;i++)
	{
		hist[i]=0;
	}
	for (int j=max(1-intY, -windowSize); j < min(h-1-intY, windowSize+1); ++j) {
		for (int i=max(1-intX, -windowSize); i < min(w-1-intX, windowSize+1); ++i) {
			float dx = i + intX - floatX;
			float dy = j + intY - floatY;
			float r2 = dx*dx + dy*dy;
			if (r2 >= (windowSize * windowSize) + 0.5) { // only compute within circle
				continue;
			}
			float weight = exp(-r2 / (2*sigmaW*sigmaW));
			float magnitude = gradImage[((intX + i) + (intY + j) * w) * 2];
			float angle = gradImage[((intX + i) + (intY + j) * w) * 2 + 1];
			int binIndex = (int)(angle / (2 * M_PI) * histSize) % histSize;
			hist[binIndex] += magnitude * weight;
		}
	}
    
	// box filter smoothing
	for (int i = 0; i < 5; ++i) {
		float prevCache;
		float currCache = hist[histSize - 1];
		float first = hist[0];
		int j;
		for (j = 0; j < (histSize - 1); ++j) {
			prevCache = currCache;
			currCache = hist[j];
			hist[j] = (prevCache + hist[j] + hist[j+1]) / 3.0f;
		}
		hist[j] = (currCache + hist[j] + first) / 3.0f;
	}
    
	// find histogram maximum
	int maxBinIndex;
	float maxBallot = -FLT_MAX;
	for (int i = 0; i < histSize; ++i) {
		if (hist[i] > maxBallot) {
			maxBallot = hist[i];
			maxBinIndex = i;
		}
	}
    
	// quadratic interpolation
	float self = hist[maxBinIndex];
	float left = (maxBinIndex == 0)? hist[histSize-1] : hist[maxBinIndex-1];
	float right = (maxBinIndex == histSize-1)? hist[0] : hist[maxBinIndex+1];
	float dx = 0.5 * (right - left);
	float dxx = (right + left - (2 * self));
	kp[index].orient = (maxBinIndex + 0.5 - (dx / dxx)) * (2 * M_PI / histSize);
#undef histSize
}

