#include <iostream>
#include <math.h>
#include "armadillo"
#include "mex.h"

using namespace arma;

void mexFunction(
        int          nlhs,
        mxArray      *plhs[],
        int          nrhs,
        const mxArray *prhs[]
        )
{
    double *ps, *pd;
    ps = mxGetPr(prhs[0]);
    pd = mxGetPr(prhs[1]);
    int nump = mxGetM(prhs[0]);
    double *pRow, *pCol;
    pRow = mxGetPr(prhs[2]);
    pCol = mxGetPr(prhs[3]);
    mat psmat(ps, nump, 2);
    mat pdmat(pd, nump, 2);

    double *mRow, *mCol;
    mRow = mxGetPr(prhs[4]);
    mCol = mxGetPr(prhs[5]);

//    mexPrintf("rows:%d", nump);
//    mexPrintf("%f, %f, %f", psmat(0), psmat(1), psmat(2));

    mat resx(*pRow, *pCol, fill::zeros);
    resx -= 1;
    mat resy = resx;
//    mat quadInd((*pRow - 1) * (*pCol - 1), 4, fill::zeros);
//    int id = 0;
    int mr = (int)*mRow;
    int mc = (int)*mCol;

//    #pragma omp for
    for (int i = 0; i < mr - 1; i++)
    {
        for (int j = 0; j < mc - 1; j++)
        {
            imat tmp = {j + i * mc, j + i * mc + 1,
                j + (i + 1) * mc + 1, j + (i + 1) * mc};
                

//            int id1 = j + i * (*pCol);
//            int id2 = j + i * (*pCol) + 1;
//            int id3 = j + (i + 1) * (*pCol) + 1;
//            int id4 = j + (i + 1) * (*pCol);
            mat A, r;
            double minx = INFINITY, miny = INFINITY, maxx = 0, maxy = 0;
            mat p(4, 3);
            for (int m = 0; m < 4; m++)
            {
                mat s = psmat.row(tmp(m));
                mat d = pdmat.row(tmp(m));
                mat ta = {{s(0), s(1), 1, 0, 0, 0, -s(0) * d(0), -s(1) * d(0)},
                         {0, 0, 0, s(0), s(1), 1, -s(0) * d(1), -s(1) * d(1)}};
                mat tr = {d(0), d(1)};
                tr = tr.t();
                A = join_vert(A, ta);
                r = join_vert(r, tr);

                // bounding box of quad
//                minx = minx < floor(s(0)) ? minx : floor(s(0));
//                maxx = maxx > ceil(s(0)) ? maxx : ceil(s(0));
//                miny = miny < floor(s(1)) ? miny : floor(s(1));
//                maxy = maxy > ceil(s(1)) ? maxy : ceil(s(1));
                minx = minx < s(0) ? minx : s(0);
                maxx = maxx > s(0) ? maxx : s(0);
                miny = miny < s(1) ? miny : s(1);
                maxy = maxy > s(1) ? maxy : s(1);

                // 4 points of src quad in fake 3d
                mat tmp = {s(0), s(1), 0};
                p.row(m) = tmp;
            }

            
            //int mySize = mxGetM(A);
            //mat b = solve(A, r);
            mexPrintf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n", 
                r(0), r(1), r(2), r(3), r(4), r(5), r(6), r(7));

            //mexPrintf("%d",b.n_elem);
            ///b =  b*r;
            mat b(3, 3, fill::eye);
            b(0) = 1;
            b(1) = 0;
            b(2) = 1;
            //mat A1 = randu<mat>(8,8);
            //mat b = solve(A, r);
            //mat b = inv(A1)*r;
            //mat b = randi<mat>(9,1,distr_param(1, 10));
            //b = join_vert(b, ones<mat>(1));
            //b = reshape(b, 3, 3).t();
            //mexPrintf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n",
            //          b(0), b(1), b(2), b(3), b(4), b(5), b(6), b(7), b(8));
            for (int m = minx; m < maxx; m++)
            {
                mexPrintf("%d",m);
                for (int n = miny; n < maxy; n++)
                {
                    // decide if the pixel is within the quad
                    vec sf(4, fill::zeros);

                    for (int k = 0; k < 4; k++)
                    {
                        mat cp = {(double)m, (double)n, 0};
                        mat v1 = p.row((k + 1) % 4) - p.row(k);
                        mat v2 = cp - p.row(k);
                        cp = cross(v1, v2);
                        sf(k) = cp(2) > 0 ? 1 : 0;
                    }

                    sf = sf - sf(0);
                    if (any(sf))
                    {
                        continue;
                    }

                    mat x = {(double)m, (double)n, 1};
                    x = x.t();
                    mat t = b * x;
                    t = t / t(2);
                    resx(n, m) = t(0);
                    resy(n, m) = t(1);
                }
            }
        }
    }
    plhs[0] = mxCreateNumericMatrix(2, (*pRow) * (*pCol), mxDOUBLE_CLASS, mxREAL);
    double *pres = (double*)mxGetData(plhs[0]);
    for (int i = 0; i < (*pRow) * (*pCol); i++)
    {
        pres[2 * i] = resx(i);
        pres[2 * i + 1] = resy(i);
    }

//    double *pvert, *pvp; // pointer of vertex and view point
//    unsigned int *pface; // pointer of faces
//    pvert = mxGetPr(prhs[0]);
//    int numv = mxGetM(prhs[0]);
//    pface = (unsigned int*)mxGetData(prhs[1]);
//    int numf = mxGetM(prhs[1]);
//    pvp = mxGetPr(prhs[2]);
//    mat vmat(pvert, numv, 3);
//    umat fmat((arma::u64 *)pface, numf, 3);
//    mat vvec(pvp, 1, 3);
//    // compute normal of each face
//    mat nmat(numf, 3);
//    #pragma omp for
//    for (int i = 0; i < numf; i++)
//    {
//        mat v1 = vmat.row(fmat(i, 2)) - vmat.row(fmat(i, 0));
//        mat v2 = vmat.row(fmat(i, 1)) - vmat.row(fmat(i, 0));
//        mat nv = arma::cross(v1, v2);
//        nmat.row(i) = nv;
//    }
//    // compute dot product
//    std::unordered_map<int, double> fmap; // first: numv * small_v_idx + large_v_idx, second: flag value
//    std::vector<int> bvec; // boundary
//    std::vector<double> tvec; // length of veiw ray
//    std::vector<mat> dvec; // direction of view ray

//    std::vector<std::vector<double> > fdvec; // direction of each face, used in kdtree
//    for (int i = 0; i < numf; i++)
//    {
//        mat nv = nmat.row(i);
//        umat ctri = fmat.row(i);
//        std::vector<int> idvec = {0, 1, 1, 2, 2, 0};
//        for (int j = 0; j < 6; j+=2)
//        {
//            int e1 = ctri(idvec[j]);
//            int e2 = ctri(idvec[j + 1]);
//            mat mp = (vmat.row(e1) + vmat.row(e2)) / 2;
//            mat vv = mp - vvec;
//            int idx = e1 > e2 ? (e2 * numv + e1) : (e1 * numv + e2);
//            double x1 = arma::dot(nv, vv);
//            double x2 = fmap[idx];
//            if (x1 * x2 < 0)
//            {
//                bvec.push_back(e1);
//                bvec.push_back(e2);
//                double rlen = arma::norm(vv, 2);
//                tvec.push_back(rlen);
//                dvec.push_back(vv / rlen);
//            }
//            fmap[idx] = x1;
//        }

//        mat fp = (vmat.row(ctri(0)) + vmat.row(ctri(1)) + vmat.row(ctri(2))) / 3.0;
//        fp = fp - vvec;
//        fp = fp / arma::norm(fp, 2);
//        std::vector<double> tv = arma::conv_to<std::vector<double> >::from(fp);
//        fdvec.push_back(tv);
//    }

//    // check if visible
//    // build kd tree

//    KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double>  kdtree(3, fdvec, 128);
//    kdtree.index->buildIndex();

//    std::vector<int> svec(tvec.size(), 1);
//    #pragma omp for
//    for (int i = 0; i < tvec.size(); i++)
//    {
//        double t1 = tvec[i];
//        mat d1 = dvec[i];

//        std::vector<double> qp = arma::conv_to<std::vector<double> >::from(d1);

//        size_t num_results = 128;
//        std::vector<size_t>   ret_index(num_results);
//        std::vector<double> out_dist_sqr(num_results);

//        kdtree.query(qp.data(), num_results, ret_index.data(), out_dist_sqr.data());

//        for (int k = 0; k < ret_index.size(); k++)
////        for (int j = 0; j < numf; j++)
//        {
//            int j = ret_index[k];
//            umat ctri = fmat.row(j);
//            mat v1 = vmat.row(ctri[0]);
//            mat v2 = vmat.row(ctri[1]);
//            mat v3 = vmat.row(ctri[2]);
//            float t2;
//            if (triangle_intersection(vec(v1.t()), vec(v2.t()), vec(v3.t()), vec(vvec.t()), vec(d1.t()), t2))
//            {
//                if ((t1 - t2) > 1e-3)
//                {
//                    svec[i] = 0;
//                    continue;
//                }
//            }
//        }
//    }

//    // pick the edge that is visible
//    std::vector<int> res;
//    for (int i = 0; i < svec.size(); i++)
//    {
//        if (svec[i])
//        {
//            res.push_back(bvec[i * 2]);
//            res.push_back(bvec[i * 2 + 1]);
//        }
//    }

//    // assign to output
//    plhs[0] = mxCreateNumericMatrix(2, res.size() / 2, mxINT32_CLASS, mxREAL);
//    int *pres = (int*)mxGetData(plhs[0]);
//    for (int i = 0; i < res.size(); i++)
//    {
//        pres[i] = res[i];
//    }
    return;
}