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
	float *img1 = new float[w*h];
	float *img2 = new float[w*h];
	float *img3 = new float[w*h];
	float inv255 = 1.0f / 256.0f;

	for (int i = 0; i < w*h; ++i) {
		img1[i] = p[i] * inv255;
	}

	printf("Non_Accel Version\n");
	Sift s1(img1, w, h, Accel_None, 0, 3, 3, false);
	vector<Keypoint> kps1 = s1.extract_keypoints(0.005f, 10.0f);
	Descriptor* des1 = new Descriptor[kps1.size()];
	s1.calc_kp_angles(&(kps1.front()),kps1.size());
	s1.calc_kp_descriptors(&(kps1.front()), kps1.size(), des1);
	printf("\n");
	delete[] img1;

	for (int i = 0; i < w*h; ++i) {
		img2[i] = p[i] * inv255;
	}
	printf("OMP_Accel Version\n");
	Sift s2(img2, w, h, Accel_OMP, 0, 3, 3, false);
	vector<Keypoint> kps2 = s2.extract_keypoints(0.005f, 10.0f);
	Descriptor* des2 = new Descriptor[kps2.size()];
	s2.calc_kp_angles(&(kps2.front()),kps2.size());
	s2.calc_kp_descriptors(&(kps2.front()), kps2.size(), des2);
	printf("\n");
	delete[] img2;

	for (int i = 0; i < w*h; ++i) {
		img3[i] = p[i] * inv255;
	}

	printf("OCL_Accel Version\n");
	Sift s3(img3, w, h, Accel_OCL, 0, 3, 3, false);
	vector<Keypoint> kps3 = s3.extract_keypoints(0.005f, 10.0f);
	Descriptor* des3 = new Descriptor[kps3.size()];
	s3.calc_kp_angles(&(kps3.front()),kps3.size());
	s3.calc_kp_descriptors(&(kps3.front()), kps3.size(), des3);
	printf("\n");

	delete[] img3;
	fclose(fp);
	free(p);
	return 0;
}
