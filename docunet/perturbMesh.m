function [ pmesh ] = perturbMesh(  )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
mr = 88;
mc = 68;
[Y, X] = meshgrid(mr : -1 : 1,  1 : mc);
ms = [X(:), Y(:)];
pmesh = ms;
nv = randi(20) - 1;
for k = 1 : nv
    % choose one vertex randomly
    vidx = randi(size(ms, 1));
    vtex = ms(vidx, :);
    % vector between all vertex and the selected one
    xv = bsxfun(@minus, pmesh, vtex);
    % random movement
    mv = (rand(1, 2) - 0.5) * 20;
    hxv = [xv, zeros(size(xv, 1), 1)];
    hmv = repmat([mv, 0], size(xv, 1), 1);
    d = bsxfun(@cross, hxv, hmv);
    d = abs(d(:, 3));
    d = d ./ norm(mv);
    % compute the influence map
    % wt = bsxfun(@minus, pmesh, vtex);
    % wt = sqrt(sum(wt.^2, 2));
    wt = d;
    
    curve_type = rand;
    if curve_type > 0.3
        alpha = rand * 50 + 50;
        % alpha = 1;
        wt = alpha ./ (wt + alpha);
    else
        alpha = rand + 1;
        wt = 1 - (wt / 100).^alpha;
    end
    
    msmv = bsxfun(@times, mv, wt);
    pmesh = pmesh + msmv;
end
% figure; scatter(pmesh(:, 1), pmesh(:, 2), 10, 1 : mr*mc, 'fill'); axis equal;
end

