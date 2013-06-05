#include "sift.h"
#include "pgm.h"
#include <cstdlib>
#include <cstdio>

int main(int argc, char** argv) {
	unsigned char *p;
	int w, h;
	FILE *fp = fopen(argv[1], "rb");
	load_P5_pgm(fp, &w, &h, &p);
	float *img = new float[w*h];
	float inv255 = 1.0f / 256.0f;
	for (int i = 0; i < w*h; ++i) {
		img[i] = p[i] * inv255;
	}
	// Sift s(img, w, h, Accel_OMP, 0, 3, 3, false);
	Sift s(img, w, h, Accel_OCL, 0, 3, 3, false);
	// Sift s(img, w, h, Accel_None, 0, 3, 3, false);
	vector<Keypoint> kps = s.extract_keypoints(0.005f, 10.0f);
	Descriptor* des = new Descriptor[kps.size()];
	s.calc_kp_angles(&(kps.front()), kps.size());
	s.calc_kp_descriptors(&(kps.front()), kps.size(), des);

	delete[] img;
	delete[] des;
	fclose(fp);
	free(p);
	return 0;
}
