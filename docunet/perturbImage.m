function [ pimg, fm, pv ] = perturbImage( img, pmesh )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
mr = 88; mc = 68;
tr = 352; tc = 272; mg = 4;
[mh, mw, ~] = size(img);
mx = linspace(mh, 1, mr);
my = linspace(1, mw, mc);

[my, mx] = meshgrid(mx, my);
smesh = cat(2, mx(:), my(:));

px = pmesh(:, 1);
py = pmesh(:, 2);

minx = min(px(:)); maxx = max(px(:)); miny = min(py(:)); maxy = max(py(:));
px = (px - minx) / (maxx - minx);
py = (py - miny) / (maxy - miny);
px = px * (tc - 2 * mg - 1) + 1 + mg;
py = py * (tr - 2 * mg - 1) + 1 + mg;
pmesh = cat(2, px, py);
fm = localwarp(pmesh, smesh, tr, tc, mr, mc);
fm = reshape(fm, 2, tr, tc);
fm = permute(fm, [2, 3, 1]);

% c++ and matlab 0/1 offset issue
fm = cat(1, fm(2 : end, :, :), fm(1, :, :));
fm = cat(2, fm(:, 2 : end, :), fm(:, 1, :));

pimg = imgMeshWarp(img, fm);

pv = pmesh;
pv = reshape(pv, mc, mr, 2);
pv = pv(:, end : -1 : 1, :);
pv = reshape(pv, [], 2);

fm = single(fm);

% pimg(isnan(pimg)) = 0;
end

