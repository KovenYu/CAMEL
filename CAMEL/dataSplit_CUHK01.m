function [data4train, Gal, Prb, labelGal, labelPrb, para] = dataSplit_CUHK01( feature, mode )

p = randperm(971);
view1 = feature(:, 1:1942);
view2 = feature(:, 1+1942:end);
idxTrain = [p(1:485)*2-1, p(1:485)*2];
idxTest = [p(486:end)*2-1, p(486:end)*2];
data4train = [view1(:, idxTrain), view2(:, idxTrain)];
data4test = [view1(:, idxTest), view2(:, idxTest)];

idxViewTrain = [ones(485*2,1);2*ones(485*2,1)];
idxViewTest = [ones(486*2,1);2*ones(486*2,1)];
labelTrain = [1:485,1:485,1:485,1:485]';
labelTest = [1:486,1:486,1:486,1:486]';
numViews = 2;

label = labelTest;
idxView = idxViewTest;

idxPrb = idxView == 2;
Prb = data4test(:, idxPrb);
labelPrb = label(idxPrb);

if strcmp(mode, 'single')
    idxGal = 1:486;
%     idxGal = idxGal + 972;
    labelGal = label(idxGal);
elseif strcmp(mode, 'multi')
    idxGal = idxView == 1;
    labelGal = label(idxGal);
else
    error('single or multi??')
end
Gal = data4test(:, idxGal);

para.idxViewTrain = idxViewTrain;
para.numViews = numViews;
para.idxViewGal = idxViewTest(idxGal);
para.idxViewPrb = idxViewTest(idxPrb);
para.labelTrain = labelTrain;