img = imread('/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_small_pngs/1AJMERCITY-PG1-0.png');
%img = imresize(img,[352 272]);
%img = imcrop(img,[500 772 500 852]);
pmesh = perturbMesh();
[pimg, fm, pv] = perturbImage(img, pmesh);
imwrite(img,'a.jpg');
imwrite(pimg,'b.jpg');