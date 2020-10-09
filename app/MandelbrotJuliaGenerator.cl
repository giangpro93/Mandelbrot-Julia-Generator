#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
void ComputeColor (int type, __global const double* baseColor, __global double* imageColor, double J_RE, double J_IM, int MaxIterations, double MaxLengthSquared, double realMin, double realMax, double imagMin, double imagMax, int nRows, int nCols, int nChannels) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int id = row*nCols + col;

    if (row<nRows && col<nCols) {
        double R_RE = realMin + (((double) col)/(nCols-1))*(realMax - realMin);
        double R_IM = imagMin + (((double) row)/(nRows-1))*(imagMax - imagMin);

        double S_RE = J_RE;
        double S_IM = J_IM;

        if (type == 0) {
            S_RE = R_RE;
            S_IM = R_IM;
        }

        int iteration=0;
        while (iteration < MaxIterations) {
            iteration++;

            // X = R^2 + S
            double X_RE = R_RE * R_RE - R_IM * R_IM + S_RE;
            double X_IM = 2 * R_RE * R_IM + S_IM;

            if (X_RE * X_RE + X_IM * X_IM > MaxLengthSquared) break;
            R_RE = X_RE;
            R_IM = X_IM;
        }

        if (iteration == MaxIterations) {
            for (int i=0; i<nChannels; i++) {
                imageColor[id*nChannels+i] = baseColor[i];
            }
        }
        else {
            double f = ((double) iteration)/MaxIterations;
            for (int i=0; i<nChannels; i++) {
                imageColor[id*nChannels+i] = (1.0 - f)*baseColor[3+i] + f*baseColor[6+i];
            }
        }
    }
}
