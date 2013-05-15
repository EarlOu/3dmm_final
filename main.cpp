#include "sift.h"
#include "pgm.h"
#include <cstdlib>
#include <cstdio>

int main(int argc, char** argv) {
	unsigned char *p;
	int w, h;
	FILE *fp = fopen("test.pgm", "rb");
	load_P5_pgm(fp, &w, &h, &p);
	float *img = new float[w*h];
	float inv255 = 1.0f / 256.0f;
	for (int i = 0; i < w*h; ++i) {
		img[i] = p[i] * inv255;
	}
	Sift s(img, w, h, 0, 4, 3, false, true);
	vector<Keypoint> kps = s.extract_keypoints(0.005f, 10.0f);
	delete[] img;
	fclose(fp);
	free(p);
	return 0;
}
