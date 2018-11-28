function [ pimg ] = perturbBackground( img, bgimg )
%UNTITLED2 Summary of this function goes here
mask = isnan(img);
pimg = im2double(img);
bgimg = im2double(bgimg);

bg_type = rand;

if bg_type > 0.9
    bgimg = bsxfun(@times, ones(size(pimg)), rand(1, 1, 3));
else
    [tr, tc, ~] = size(bgimg);
    [sr, sc, ~] = size(img);
    while (any([tr, tc] < [sr, sc]))
        bgimg = repmat(bgimg, 2, 2);
        [tr, tc, ~] = size(bgimg);
    end
    bgimg = bgimg(1 : sr, 1 : sc, :);
end

pimg(mask) = bgimg(mask);