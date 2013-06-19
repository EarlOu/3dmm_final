#ifndef SIFT_H
#define SIFT_H
#include <vector>
using std::vector;

struct CLStruct;

struct Keypoint {
	int ix, iy, is, o;
	float x, y, sigma, orient;
};

struct Descriptor {
	float v[128];
	void normalize();
	void regularize();
};

enum AccerType {
	Accel_None = 0,
	Accel_OMP,
	Accel_OCL
};

struct AccerModel {
	AccerType gaussian;
	AccerType dog;
	AccerType gradient;
	AccerType angle;
	AccerType descriptor;
};

static const AccerModel AccerNoneAll = {Accel_None, Accel_None, Accel_None, Accel_None, Accel_None};

class Sift {
public:
	Sift(
		float *_img, int _w, int _h,
		AccerModel acc = AccerNoneAll,
		int _octMin = -1, int _numOct = 3, int _lvPerScale = 3,
		bool _dumpImage = false
	);
	~Sift();
	const vector<Keypoint> &extract_keypoints(float mth = 0.1f, float eth = 15.0f);
	void calc_kp_angle(Keypoint &kp);
	void calc_kp_angles(Keypoint *kps, int n);
	void calc_kp_descriptor(const Keypoint &kp, Descriptor &des);
	void calc_kp_descriptors(const Keypoint *kps, int n, Descriptor *dess);
	bool dumpImage;
private:
	bool hasGaussian, hasGrads;
	AccerModel accel;
	float *img, *buffer;
	float **blurred, **dogs, *magAndThetas;
	int imgw, imgh, wmax, hmax;
	int octMin, numOct, lvPerScale;
	float sigma0;
	float eth, mth;
	int gradient_map_size;

	void init_gaussian_mem();
	void init_gaussian_first();
	void init_gaussian_build();
	void dump_gaussian_build();
	void init_gaussian_dog();
	void dump_gaussian_dog();
	void init_gaussian();

	void detect_raw_keypoints();
	void refine_keypoints();
	void build_gradient_images();

	void init_gradient_mem();
	void init_gradient_build();
	void init_gradient();
	float *get_gradient(int o);

	void calc_kp_descriptors_OCL(const struct Keypoint* kps,
		int n, struct Descriptor* dess);

	vector<Keypoint> kps;
	CLStruct *cls;
};

#endif
