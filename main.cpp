#include "sift.h"
#include "pgm.h"
#include <cstdlib>
#include <cstdio>

int main(int argc, char** argv) {
	unsigned char *p;
	int w, h;

	if (argc != 2) {
		printf("Usage: %s <file>\n", argv[0]);
		return 1;
	}

	FILE *fp = fopen(argv[1], "rb");

	if(!fp) {
		printf("Cannot open %s\n", argv[1]);
		return 1;
	}

	load_P5_pgm(fp, &w, &h, &p);
	float *img = new float[w*h];
	float inv255 = 1.0f / 255.0f;
	for (int i = 0; i < w*h; ++i) {
		img[i] = p[i] * inv255;
	}


	AccerModel a1 = {Accel_None, Accel_None, Accel_None, Accel_None, Accel_None};
	AccerModel a2 = {Accel_OCL, Accel_OMP, Accel_OCL, Accel_None, Accel_OMP};

	printf("Not accelerated:\n");
	Sift s1(img, w, h, a1, 0, 4, 3, false);
	vector<Keypoint> kps1 = s1.extract_keypoints(0.005f, 10.0f);
	Descriptor* des1 = new Descriptor[kps1.size()];
	s1.calc_kp_angles(kps1.data(),kps1.size());
	s1.calc_kp_descriptors(kps1.data(), kps1.size(), des1);
	printf("\n");

	printf("With base acceleration model:\n");
	Sift s2(img, w, h, a2, 0, 4, 3, false);
	vector<Keypoint> kps2 = s2.extract_keypoints(0.005f, 10.0f);
	Descriptor* des2 = new Descriptor[kps2.size()];
	s2.calc_kp_angles(kps2.data(),kps2.size());
	s2.calc_kp_descriptors(kps2.data(), kps2.size(), des2);
	printf("\n");

	fclose(fp);
	free(p);
	return 0;
}
