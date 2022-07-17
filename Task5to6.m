% Task5to6
clear; clc, close all

% Task 5: Robust method --------------------------

% Step-1: Load input image
adres = 'Assignment_GT';
fname = 'IMG_15_GT.png';

GT = imread(fullfile(adres,fname));

figure, imshow(GT*125)
title('Original Image times 125')
saveas(gcf,fullfile('output1',[fname(1:end-4),'_origx125','.png'])) % save the image

% To visualise image
I = rgb2gray(GT);

BW = imbinarize(I);
CC = bwconncomp(BW);
L = labelmatrix(CC);

L_GT = label2rgb(L, 'prism','k','shuffle');
figure, imshow(L_GT)
title('Original Image with Prism Colormap')
saveas(gcf,fullfile('output1',[fname(1:end-4),'_origPrism','.png'])) % save the image

% Step-2: Covert image to grayscale
I = L_GT;
I_gray = rgb2gray(I);
figure, imshow(I_gray)
title('Grey-scale Image')
saveas(gcf,fullfile('output1',[fname(1:end-4),'_grey','.png'])) % save the image

% Step-3: Rescale image
I_gray_sc = imresize(I_gray,512/size(I_gray,1));
figure, imshow(I_gray_sc)
title('Resized Grey-scale Image')
saveas(gcf,fullfile('output1',[fname(1:end-4),'_resized','.png'])) % save the image

% Step-4: Produce histogram before enhancing
figure
histogram(double(I_gray_sc(:)),64)
title('Histogram of the Resized Grey-scale Image')
saveas(gcf,fullfile('output1',[fname(1:end-4),'_hist1','.png'])) % save the image

%% Step-5: Enhance image before binarisation
Idouble = im2double(I_gray_sc);
avg = mean2(Idouble);
sigma = std2(Idouble);
n = avg/sigma*0.99;% n must be smaller than avg/sigma
I_gray_sc_e = imadjust(I_gray_sc,[avg-n*sigma avg+n*sigma],[]);

figure, imshow(I_gray_sc_e)
title('Enhanced of the Resized Grey-scale Image')
saveas(gcf,fullfile('output1',[fname(1:end-4),'_enhanced','.png'])) % save the image

%% Step-6: Histogram after enhancement
figure, histogram(double(I_gray_sc_e(:)),64)
title('Histogram of the Enhanced Resized Grey-scale Image')
saveas(gcf,fullfile('output1',[fname(1:end-4),'_hist2','.png'])) % save the image

%% Step-7: Image Binarisation
I_bw = imbinarize(I_gray_sc_e);
I_bw = bwareaopen(I_bw,100,8);% remove small objects
figure, imshow(I_bw)
title('Binarised Enhanced Resized Grey-scale Image')
saveas(gcf,fullfile('output1',[fname(1:end-4),'_bin','.png'])) % save the image

% Task 2: Edge detection ------------------------
e1 = edge(I_bw,'sobel');
figure, imshow(e1)
title('Detected Edges of the Binarised Enhanced Resized Grey-scale Image')
saveas(gcf,fullfile('output1',[fname(1:end-4),'_edges','.png'])) % save the image

% Task 3: Simple segmentation --------------------
% Ref: https://www.mathworks.com/help/images/detecting-a-cell-using-image-segmentation.html
se90 = strel('line',3,90);
se0 = strel('line',3,0);

BWsdil = imdilate(e1,[se90 se0],'full');
figure, imshow(BWsdil)
title('Dilated Gradient Mask')
saveas(gcf,fullfile('output1',[fname(1:end-4),'_d','.png'])) % save the image

BWdfill = imfillb(BWsdil);
figure, imshow(BWdfill)
title('Binary Image with Filled Holes')
saveas(gcf,fullfile('output1',[fname(1:end-4),'_fill','.png'])) % save the image

%% Task 4: Object Recognition --------------------
m = regionprops('table',BWdfill,'PixelList','Solidity','Area');

rgb = zeros(size(BWdfill,1),size(BWdfill,2),3,'uint8');
nObj = size(m,1);% number of objects
for o = 1 : nObj
    i = m.PixelList{o};
    A = m.Area(o);
    if (A < 3000) || (m.Solidity(o) < 0.45) % bacteria
        rgb1 = [0 0 1]*255;% blue
    else % blood cell
        rgb1 = [1 0 0]*255;% red
    end
    for k = 1 : size(i,1)
        rgb(i(k,2), i(k,1), :) = rgb1;
    end
end

figure, imshow(rgb)
title('Recognised objects: Blood cells in Red, Bacteria in Blue')
saveas(gcf,fullfile('output1',[fname(1:end-4),'_rgb','.png'])) % save the image

%% Task 6: Performance evaluation -----------------

% after running the results gathered
blbc = [...
    10 0
    8 0
    0 1
    6 3
    2 2
    4 2
    6 2
    9 4
    14 6
    7 5
    7 1
    14 0
    6 1
    19 0
    3 1];

blbcp = [...
    10 0
    8 0
    0 1
    6 3
    2 1
    4 0
    6 2
    9 2
    13 2
    7 3
    7 1
    14 0
    6 1
    19 0
    3 1];

TP = sum(blbcp(:,1))
TN = sum(blbcp(:,2))
FP = sum(blbc(:,2)-blbcp(:,2))
FN = sum(blbc(:,1)-blbcp(:,1))

diceScore = 2*TP/(2*TP + FP + FN)
precision = TP/(TP + FP)
recall = TP/(TP + FN)


%-------------------------------------------------------------------------------
function bw_filled=imfillb(bw) % Fill holes on the border of a binary image

% add a column of white pixels on the left and a row of white pixels on the top
bw_a = padarray(bw,[1 1],1,'pre');
bw_a_filled = imfill(bw_a,'holes');
bw_a_filled = bw_a_filled(2:end,2:end);

% fill against the top and the right border
bw_b = padarray(padarray(bw,[1 0],1,'pre'),[0 1],1,'post');
bw_b_filled = imfill(bw_b,'holes');
bw_b_filled = bw_b_filled(2:end,1:end-1);

% fill against the right and bottom borders
bw_c = padarray(bw,[1 1],1,'post');
bw_c_filled = imfill(bw_c,'holes');
bw_c_filled = bw_c_filled(1:end-1,1:end-1);

% fill against the bottom and left borders
bw_d = padarray(padarray(bw,[1 0],1,'post'),[0 1],1,'pre');
bw_d_filled = imfill(bw_d,'holes');
bw_d_filled = bw_d_filled(1:end-1,2:end);

% The last step is then to "logical OR" all these images together
bw_filled = bw_a_filled | bw_b_filled | bw_c_filled | bw_d_filled;
end
