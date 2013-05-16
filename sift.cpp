#include "sift.h"
#include "utils.h"
#include "pgm.h"

#include <cstring>
#include <cmath>
#include <cstdio>

Sift::Sift(
	float *_img, int _w, int _h,
	int _octMin, int _numOct, int _lvPerScale,
	bool _useCL, bool _dumpImage
)
{
	img = _img;
	wmax = w = _w;
	hmax = h = _h;
	octMin = _octMin;
	if (octMin < 0) {
		wmax <<= -octMin;
		hmax <<= -octMin;
	} else if (octMin > 0) {
		wmax >>= octMin;
		hmax >>= octMin;
	}
	numOct = _numOct;
	lvPerScale = _lvPerScale;
	sigma0 = 1.6f * powf(2.0f, 1.0f/lvPerScale);
	hasGaussian = hasGrads = false;
	useCL = _useCL;
	dumpImage = _dumpImage;
	init_gaussian();
}

Sift::~Sift()
{
	if (hasGaussian) {
		delete[] buffer;
		for (int i = 0; i < numOct; ++i) {
			delete[] blurred[i];
			delete[] dogs[i];
		}

		delete[] blurred;
		delete[] dogs;
	}
	if (hasGrads) {
		delete[] magAndThetas;
	}
}

void Sift::init_gaussian_mem()
{
	int wtmp = wmax;
	int htmp = hmax;
	buffer = new float[wtmp*htmp];
	blurred = new float*[numOct];
	dogs = new float*[numOct];
	for (int i = 0; i < numOct; ++i) {
		blurred[i] = new float[wtmp * htmp * (lvPerScale+3)];
		dogs[i] = new float[wtmp * htmp * (lvPerScale+2)];
		wtmp >>= 1;
		htmp >>= 1;
	}
}

void Sift::init_gaussian_first()
{
	if(octMin < 0) {
		upSample2(blurred[0], img, buffer, w, h);
		for (int o = 1; o < -octMin; ++o) {
			upSample2(blurred[0], blurred[0], buffer, w<<o, h<<o);
		}
	} else if(octMin > 0) {
		downSample(blurred[0], img, wmax, hmax, 1<<octMin);
	} else {
		memcpy(blurred[0], img, w*h*sizeof(float));
	}
}

void Sift::init_gaussian_build()
{
	int wtmp = wmax;
	int htmp = hmax;

	float sigmak = powf(2.0f, 1.0f/lvPerScale);
	float dsigma0 = sigma0 * sqrtf(1.0f - 1.0f/(sigmak*sigmak));
	float dsigmar = sigmak;
	for (int o = 0; o < numOct; ++o) {
		float sigma = dsigma0;
		int imsiz = wtmp*htmp;
		for(int s = 0 ; s <= lvPerScale+1; ++s) {
			gaussian_blur(
				blurred[o]+imsiz*(s+1), blurred[o]+imsiz*s,
				buffer, wtmp, htmp, sigma
			);
			sigma *= dsigmar;
		}

		if (o != numOct-1) {
			downSample(blurred[o+1], blurred[o]+imsiz*lvPerScale, wtmp, htmp, 2);
		}

		wtmp >>= 1;
		htmp >>= 1;
	}

	if (dumpImage) {
		dump_gaussian_build();
	}
}

void Sift::init_gaussian_dog()
{
	int wtmp = wmax;
	int htmp = hmax;
	for (int o = 0; o < numOct; ++o) {
		diff(dogs[o], blurred[o], lvPerScale+3, wtmp, htmp);
		wtmp >>= 1;
		htmp >>= 1;
	}

	if (dumpImage) {
		dump_gaussian_dog();
	}
}

void Sift::dump_gaussian_build()
{
	int wtmp = wmax;
	int htmp = hmax;
	unsigned char *wrbuf = new unsigned char[wmax*hmax];
	for (int o = 0; o < numOct; ++o) {
		int imgSiz = wtmp*htmp;
		for(int s = 0 ; s <= lvPerScale+2; ++s) {
			char buf[128];
			sprintf(buf, "g_%d_%d.pgm", o, s);
			FILE *fp = fopen(buf, "wb");
			float *based = blurred[o] + s*imgSiz;
			for (int i = 0; i < imgSiz; ++i) {
				wrbuf[i] = based[i]*256.0f;
			}
			save_P5_pgm(fp, wtmp, htmp, wrbuf);
			fclose(fp);
		}
		wtmp >>= 1;
		htmp >>= 1;
	}
	delete[] wrbuf;
}

void Sift::dump_gaussian_dog()
{
	int wtmp = wmax;
	int htmp = hmax;
	unsigned char *wrbuf = new unsigned char[wmax*hmax];
	for (int o = 0; o < numOct; ++o) {
		int imgSiz = wtmp*htmp;
		for(int s = 0 ; s < lvPerScale+2; ++s) {
			char buf[128];
			sprintf(buf, "d_%d_%d.pgm", o, s);
			FILE *fp = fopen(buf, "wb");
			float *based = dogs[o] + s*imgSiz;
			for (int i = 0; i < imgSiz; ++i) {
				wrbuf[i] = based[i]*256.0f + 128.0f;
			}
			save_P5_pgm(fp, wtmp, htmp, wrbuf);
			fclose(fp);
		}
		wtmp >>= 1;
		htmp >>= 1;
	}
	delete[] wrbuf;
}

void Sift::init_gaussian()
{
	hasGaussian = true;
	init_gaussian_mem();
	init_gaussian_first();
	init_gaussian_build();
	init_gaussian_dog();

}

void Sift::detect_raw_keypoints()
{
	int wtmp = wmax;
	int htmp = hmax;
	for (int o = 0; o < numOct; ++o) {
		// detect feature points (search for 26 neighboring points)
		int image_size = wtmp*htmp;
		int line_size = wtmp;
		for (int s = 1; s < lvPerScale+1; ++s) {
			for (int i = 1; i < (htmp - 1); ++i) {
				for (int j = 1; j < (wtmp - 1); ++j) {
					float *imgbase = dogs[o]+(s*wtmp*htmp);
#define CHECK_EXTREMA(OPER, THRES, PTR, IMAGE_SIZE, LINE_SIZE) ( \
	(*(PTR) OPER THRES)                               &&         \
	(*(PTR) OPER *(PTR - LINE_SIZE - 1))              &&         \
	(*(PTR) OPER *(PTR - LINE_SIZE))                  &&         \
	(*(PTR) OPER *(PTR - LINE_SIZE + 1))              &&         \
	(*(PTR) OPER *(PTR - 1))                          &&         \
	(*(PTR) OPER *(PTR + 1))                          &&         \
	(*(PTR) OPER *(PTR + LINE_SIZE - 1))              &&         \
	(*(PTR) OPER *(PTR + LINE_SIZE))                  &&         \
	(*(PTR) OPER *(PTR + LINE_SIZE + 1))              &&         \
	(*(PTR) OPER *(PTR - IMAGE_SIZE - LINE_SIZE - 1)) &&         \
	(*(PTR) OPER *(PTR - IMAGE_SIZE - LINE_SIZE))     &&         \
	(*(PTR) OPER *(PTR - IMAGE_SIZE - LINE_SIZE + 1)) &&         \
	(*(PTR) OPER *(PTR - IMAGE_SIZE - 1))             &&         \
	(*(PTR) OPER *(PTR - IMAGE_SIZE + 1))             &&         \
	(*(PTR) OPER *(PTR - IMAGE_SIZE + LINE_SIZE - 1)) &&         \
	(*(PTR) OPER *(PTR - IMAGE_SIZE + LINE_SIZE))     &&         \
	(*(PTR) OPER *(PTR - IMAGE_SIZE + LINE_SIZE + 1)) &&         \
	(*(PTR) OPER *(PTR + IMAGE_SIZE - LINE_SIZE - 1)) &&         \
	(*(PTR) OPER *(PTR + IMAGE_SIZE - LINE_SIZE))     &&         \
	(*(PTR) OPER *(PTR + IMAGE_SIZE - LINE_SIZE + 1)) &&         \
	(*(PTR) OPER *(PTR + IMAGE_SIZE - 1))             &&         \
	(*(PTR) OPER *(PTR + IMAGE_SIZE + 1))             &&         \
	(*(PTR) OPER *(PTR + IMAGE_SIZE + LINE_SIZE - 1)) &&         \
	(*(PTR) OPER *(PTR + IMAGE_SIZE + LINE_SIZE))     &&         \
	(*(PTR) OPER *(PTR + IMAGE_SIZE + LINE_SIZE + 1)))

					float *ptr = imgbase + i*wtmp + j;
					if (CHECK_EXTREMA(>,  mth, ptr, image_size, line_size) ||
						CHECK_EXTREMA(<, -mth, ptr, image_size, line_size)) {
						Keypoint key;
						key.o = o;
						key.ix = j;
						key.iy = i;
						key.is = s-1;
						kps.push_back(key);
					}
				}
			}
		}
		wtmp >>= 1;
		htmp >>= 1;
	}
	printf("%lu raw kps\n", kps.size());
}

void Sift::refine_keypoints()
{
	int o, s, w, h, x, y;
	float* point; // position of the key point
	float delta[3]; // displacement of point
	float gradient[3]; // 1st-order derivative
	float hessian[9]; // 2nd-order derivative
	float hessianInv[9]; // inverse of hessian

	int dst = 0;
	for (int i = 0; i < kps.size(); ++i) {
		o = kps[i].o;
		x = kps[i].ix;
		y = kps[i].iy;
		s = kps[i].is;
		w = wmax >> o;
		h = hmax >> o;
		point = dogs[o] + (s+1)*w*h + (y*w+x);

		build_gradient(gradient, point, w, h);
		build_hessian(hessian, point, w, h);
		inv_3d_matrix(hessianInv, hessian);

		// delta = inv((d/dr)^2 D) * (dD/dr)
		// r' = r - delta
		matrix_multiply(delta, hessianInv, gradient);

		float period = powf(2.0f, o+octMin);
		kps[i].x = (x - delta[0]) * period;
		kps[i].y = (y - delta[1]) * period;
		float s_unscale = s - delta[2];
		kps[i].sigma = 1.6 * powf(2.0f, o + (s_unscale / lvPerScale));
		if (kps[i].sigma < 1) {
			kps[i].sigma = 1;
		}


		// reject edge response, [ dxx dxy ]
		//                       [ dyx dyy ], 'edge' if (trace^2/det) > (th+1)^2/th
		float trace = hessian[0] + hessian[4];
		float det = hessian[0] * hessian[4] - hessian[1] * hessian[1];
		if ((trace*trace/det)<((eth+1)*(eth+1)/eth)) {
			kps[dst] = kps[i];
			++dst;
		}
	}
	kps.resize(dst);
	printf("%lu kps after refine\n", kps.size());
}

const vector<Keypoint> &Sift::extract_keypoints(float _mth, float _eth)
{
	mth = _mth;
	eth = _eth;
	detect_raw_keypoints();
	refine_keypoints();
	return kps;
}

void Sift::init_gradient_mem()
{
	int wtmp = wmax;
	int htmp = hmax;
	magAndThetas = new float*[numOct];
	for (int i = 0; i < numOct; ++i) {
		magAndThetas[i] = new float[2*wtmp*htmp*lvPerScale];
		wtmp >>= 1;
		htmp >>= 1;
	}
}

void Sift::init_gradient_build()
{
	int wtmp = wmax;
	int htmp = hmax;
	for (int o = 0; o < numOct; ++o) {
		int imgsiz = wtmp*htmp;
		build_gradient_map(magAndThetas[o], blurred[o]+imgsiz, lvPerScale, wtmp, htmp);
		wtmp >>= 1;
		htmp >>= 1;
	}
}

void Sift::init_gradient()
{
	hasGrads = true;
	init_gradient_mem();
	init_gradient_build();
}

void Sift::calc_kp_angle(Keypoint &kp)
{
	if (!hasGrads) {
		init_gradient();
	}

	int o = kp.o;
	int s = kp.is;
	float period = powf(2.0f, o);
	float sigmaW = 1.5f * (kp.sigma / period);
	int windowSize = 3 * sigmaW;
	if (windowSize <= 0) {
		windowSize = 1;
	}
	int w = wmax>>o;
	int h = hmax>>o;
	float* gradImage = magAndThetas[o] + s*w*h;
	float floatX = kp.x / period;
	float floatY = kp.y / period;
	int intX = (int)(floatX + 0.5);
	int intY = (int)(floatY + 0.5);

	if (intX < 1 || intX >= w || intY < 1 || intY >= h) {
		return;
	}

	// compute histogram
	const int histSize = 36;
	float hist[histSize];
	memset(hist, 0, histSize * sizeof(float));
	for (int j=MAX(1-intY, -windowSize); j < MIN(h-1-intY, windowSize+1); ++j) {
		for (int i=MAX(1-intX, -windowSize); i < MIN(w-1-intX, windowSize+1); ++i) {
			float dx = i + intX - floatX;
			float dy = j + intY - floatY;
			float r2 = dx*dx + dy*dy;
			if (r2 >= (windowSize * windowSize) + 0.5) { // only compute within circle
				continue;
			}
			float weight = expf(-r2 / (2*sigmaW*sigmaW));
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
	float maxBallot = 0.0f;
	for (int i = 0; i < histSize; ++i) {
		if (hist[i] > maxBallot) {
			maxBallot = hist[i];
			maxBinIndex = i;
		}
	}

	// quadratic interpolation
	float self = hist[maxBinIndex];
	float left = (maxBinIndex == 0)? hist[histSize-1] : hist[maxBinIndex-1];
	float right = (maxBinIndex == histSize)? hist[0] : hist[maxBinIndex+1];
	float dx = 0.5 * (right - left);
	float dxx = (right + left - (2 * self));
	kp.orient = (maxBinIndex + 0.5 - (dx / dxx)) * (2 * M_PI / histSize);
}

void Sift::calc_kp_angles(Keypoint *kps, int n)
{
	for (int i = 0; i < n; ++i) {
		calc_kp_angle(*kps);
		++kps;
	}
}


void Sift::calc_kp_descriptor(const Keypoint &kp, Descriptor &des)
{
	if (!hasGrads) {
		init_gradient();
	}
}

void Sift::calc_kp_descriptors(const Keypoint *kps, int n, Descriptor *dess)
{
	for (int i = 0; i < n; ++i) {
		calc_kp_descriptor(*kps, *dess);
		++kps;
		++dess;
	}
}

