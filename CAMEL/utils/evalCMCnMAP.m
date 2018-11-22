function [ CMC, mAP ] = evalCMCnMAP( dist, para )

IdGal = para.labelTest;
nQuery = para.numTotalImgQuery;
labelQuery = para.labelQuery;
viewQuery = para.idxViewQuery;
viewGal = para.idxViewTest;
numTotalImgGal = para.numTotalImgTest;

junk0 = find(IdGal == -1);
ap = zeros(nQuery, 1);
CMC = zeros(numTotalImgGal, nQuery);

for i = 1:nQuery
    score = dist(:, i);
    q_label = labelQuery(i);
    q_cam = viewQuery(i);
    pos = find(IdGal == q_label);
    pos2 = viewGal(pos) ~= q_cam;
    good_image = pos(pos2);
    pos3 = viewGal(pos) == q_cam;
    junk = pos(pos3);
    junk_image = [junk0; junk];
    [~, index] = sort(score, 'ascend');
    [ap(i), CMC(:, i)] = compute_AP(good_image, junk_image, index);
end
CMC = sum(CMC, 2)./nQuery;
CMC = CMC';
mAP = sum(ap)/length(ap);

end