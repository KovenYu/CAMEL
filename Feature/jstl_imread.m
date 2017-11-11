function input_data = jstl_imread( dir, img )

if nargin == 1
    im = imread(dir);
else % img exists
    im = img;
end
im = imresize(im,[144,56]);
% im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
% im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im);  % convert from uint8 to single
% im_data(:,:,1) = im_data(:,:,1) - 102;
% im_data(:,:,2) = im_data(:,:,2) - 102;
% im_data(:,:,3) = im_data(:,:,3) - 101;

im_data(:,:,1) = im_data(:,:,1) - 101;
im_data(:,:,2) = im_data(:,:,2) - 102;
im_data(:,:,3) = im_data(:,:,3) - 102;
input_data = im_data;


end