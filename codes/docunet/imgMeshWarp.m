function res = imgMeshWarp( img, flowmap )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
img = im2double(img);
rR = img(:, :, 1);
rG = img(:, :, 2);
rB = img(:, :, 3);
fx = flowmap(:, :, 1); fy = flowmap(:, :, 2);
VqR = interp2(rR, fx(:), fy(:));
VqG = interp2(rG, fx(:), fy(:));
VqB = interp2(rB, fx(:), fy(:));
res = cat(3, VqR, VqG, VqB);
res = reshape(res, size(flowmap, 1), size(flowmap, 2), []);

end

