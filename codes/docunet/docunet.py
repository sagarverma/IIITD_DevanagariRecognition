import cv2
import random
import numpy.matlib
import numpy as np
from scipy import interpolate
np.set_printoptions(suppress=True) 

def perturbMesh(img):
    rows,cols = img.shape[0],img.shape[1]
    mr = 89;
    mc = 69;
    Y = np.arange(1,mr+1)
    X = np.arange(1,mc+1)
    Y = np.flip(Y,axis=0)

    mesh = np.meshgrid(X,Y)
    mesh = np.array(mesh)
    ms = []
    for i in range(len(Y)):
        for j in range(1,70):
            ms.append([j,Y[i]])
    ms = np.array(ms)
    pmesh = ms
    nv = random.randint(5,25) - 1;
    for k in range(1,nv):
        vidx = random.randint(1,len(ms))
        vtex = ms[vidx]
        xv = pmesh - vtex
        mv = (np.random.rand(1, 2) - 0.5) * 15
       
        zeros = np.zeros((xv.shape[0],1))
        
        hxv = np.concatenate((xv, zeros), axis=1)
        
        
        hmv = np.matlib.repmat([mv, 0], xv.shape[0],1)
        d = np.cross(hxv,hmv)
        d = abs(d)
        d = d / np.linalg.norm(mv)
        d = d[:,:1]

        wt = d
        rand = random.uniform(0,1)
        curve_type = rand
        if curve_type > 0.3 :
            alpha = rand * 45 + 55
            wt = alpha / (wt + alpha)
        else:
            alpha = rand + 1
            wt = 1 - (wt / 100)**alpha
        
        msmv = np.multiply(mv,wt)
        
        pmesh = pmesh + msmv
    print(pmesh.shape)
    return pmesh
def localWarp(pmesh,smesh,tr,tc,mr,mc):

    INFINITY =  1000000000000000000000000000.00
    nump = pmesh.size

    pRow = tr
    pCol = tc
    
    psmat = np.reshape(pmesh,(int(nump/2),2))
    pdmat = np.reshape(smesh,(int(nump/2),2))
    resx = np.zeros((pRow,pCol))
    resx -= 1

    for i in range(mr-1):
        for j in range(mc-1):   
            tmp = [j + i * mc, j + i * mc + 1,
                j + (i + 1) * mc + 1, j + (i + 1) * mc]
            tmp = np.array(tmp)
            print(tmp)
            minx = INFINITY
            miny = INFINITY
            maxx = 0
            maxy = 0

            p = []
            
            for m in range(4):

                s = psmat[tmp[m]]
                d = pdmat[tmp[m]]
                ta = [[s[0], s[1], 1, 0, 0, 0, -s[0] * d[0], -s[1] * d[0]],
                         [0, 0, 0, s[0], s[1], 1, -s[0] * d[1], -s[1] * d[1]]]
                tr = [d[0], d[1]]
                ta = np.array(ta)
                tr = np.array(tr)

                tr = np.transpose(tr)
                if(m==0):
                    A = ta
                    r = tr
                else:
                    A = np.concatenate((A,ta), axis=1)
                    r = np.concatenate((r,tr), axis=0)


                minx = minx if minx < s[0] else s[0]
                maxx = maxx if maxx > s[0] else s[0]
                miny = miny if miny < s[1] else s[1]
                maxy = maxy if maxy > s[1] else s[1]

                tmp1 = [s[0], s[1], 0]
                p.append(tmp1)
            p = np.array(p)
            A = A.reshape(8,8)
            print(A)
            print(np.linalg.det(A))
            print(r)
            A = np.random.rand(8,8)
            b = np.linalg.inv(A) * np.transpose(r)
            b = np.concatenate((b,np.ones(1)),axis=1)

            b = np.transpose(np.reshape(b,(3, 3)))

            for m in range(minx,maxx):
                for n in range(miny,maxy):
                    
                    sf = np.zeros((1,4))

                    for k in range(4):
                        cp = np.array([double(m), double(n), 0])
                        v1 = p[(k + 1) % 4] - p[k]
                        v2 = cp - p[k]                        
                        cp = np.cross(v1, v2);
                        sf[k] = 1 if cp[2] > 0 else 0

                    sf = sf - sf[0];
                    if (np.any(sf)):
                        continue

                    x = np.transpose(np.array([double(m), double(n), 1]))
                    t = b * x;
                    t = t / t[2]
                    resx[n, m] = t[0]
                    resy[n, m] = t[1]
    pres = np.array((pRow*pCol,1),dtype=np.float32)
    for i in range(pRow*pCol):
        pres[2 * i] = resx[i]
        pres[2 * i + 1] = resy[i]
    pres = np.reshape(pres,(pRow,pCol))
    return pres

def imgMeshWarp(img, flowmap):
    rR = img[:, :, 1]
    rG = img[:, :, 2]
    rB = img[:, :, 3]
    fx = flowmap[:, :, 1] 
    fy = flowmap[:, :, 2]
    VqR = interpolate.interp2(rR, fx, fy)
    VqG = interpolate.interp2(rG, fx, fy)
    VqB = interpolate.interp2(rB, fx, fy)
    res = np.concatenate((VqR, VqG, VqB),axis = 2)
    res = np.reshape(res, flowmap.shape[0], flowmap.shape[1], 3)

def perturbImage(img, pmesh):
    mr = 89
    mc = 69
    tr = 100
    tc = 100 
    mg = 5
    [mh, mw,c] = img.shape
    mx = np.linspace(mh, 1, mr)
    my = np.linspace(1, mw, mc)
    [my,mx] = np.meshgrid(mx,my)
    smesh = np.concatenate((mx,my),axis=1)
    px = pmesh[:, 0]
    py = pmesh[:, 1]
    minx = np.min(px)
    maxx = np.max(px)
    miny = np.min(py)
    maxy = np.max(py)
    px = (px - minx) / (maxx - minx)
    py = (py - miny) / (maxy - miny)
    px = px * (tc - 2 * mg - 1) + 1 + mg
    py = py * (tr - 2 * mg - 1) + 1 + mg
    pmesh = np.concatenate((px, py),axis=0);
    print(pmesh.shape)
    fm = localWarp(pmesh, smesh, tr, tc, mr, mc);
    fm = np.reshape(fm, 2, tr, tc);
    fm = np.permute(fm, [2, 3, 1]);

    fm = np.concatenate((fm[2 : end, :, :], fm[1, :, :]),axis=0)
    fm = np.concatenate((fm[:, 2 : end, :], fm[:, 1, :]),axis = 1)

    pimg = imgMeshWarp(img, fm)
    pv = pmesh
    pv = np.reshape(pv, mc, mr, 2)
    pv = pv[:, end : -1 : 1, :]
    pv = np.reshape(pv, (-1, 2))

    return pimg

def create_dataset():
    img = cv2.imread("test.jpg")
    print img.shape
    mesh = perturbMesh(img)
    result = perturbImage(img,mesh)
    cv2.imshow(result)
    cv2.waitKey(0)

create_dataset()