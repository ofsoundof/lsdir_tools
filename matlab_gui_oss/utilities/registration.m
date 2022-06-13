function [] = registration(username)
%
global results
global imgDataPath

absImgDataPath = what(imgDataPath);
imgDataPath = absImgDataPath.path;
conf = find_img_path(imgDataPath);

% 排序
extract_numbers = [];
for i = 1:length(conf)
    [startIndex,endIndex] = regexp(conf(i).folder,'_');
    extract_numbers = [extract_numbers,str2num(conf(i).folder(endIndex(end)+1:end))];
end
[sorted_numbers, ind] = sort(extract_numbers);
sorted_conf = conf(ind);
conf = sorted_conf;

disp(['Total number of images:', num2str(length(conf))])

conf(1).is_deleted = 2;

save(fullfile(results,['Subject#_',username,'_double_check.mat']),'conf');
end

function imgs = find_img_path(imgPath)
%   recursively find all of the png images under a path.
    imgs = [];
    imgDataDir  = dir(imgPath);             % 遍历所有文件
    for i = 1:length(imgDataDir)
        path = imgDataDir(i);
        if path.isdir && ~(isequal(path.name,'.')||isequal(path.name,'..')) % 去除系统自带的两个隐文件夹
            imgs_sub = find_img_path(fullfile(path.folder, path.name));
            if ~isempty(imgs_sub)
                imgs = [imgs; imgs_sub];
            end
        elseif ~path.isdir && endsWith(path.name, '.png')
            path.is_deleted = 0;
            path.time_operation = [];
            imgs = [imgs; path];
        else
            continue;
        end
    end
    disp(length(imgs))
end

        % imgDataPath = '/home/thor/projects/scrapy_exercise/part1/flickr2k/x4/l0';
    %     if endsWith(imgDataPath, 'l0') || endsWith(imgDataPath, 'lother')
    %         imgDir = dir(fullfile(imgDataPath, imgDataDir(i).name, '*.png'));
    %     elseif endsWith(imgDataPath, 'x4')
    %         imgDir = dir(fullfile(imgDataPath, imgDataDir(i).name, '*/*.png'));
    %     elseif endsWith(imgDataPath, 'flickr2k') || endsWith(imgDataPath, 'flickr_tag') ||...
    %             endsWith(imgDataPath, 'imagenet') || endsWith(imgDataPath, 'imagenet_21k')
    %         imgDir = dir(fullfile(imgDataPath, imgDataDir(i).name, '*/*/*.png'));
    %     else
    %         imgDir = dir(fullfile(imgDataPath, imgDataDir(i).name,'*/*/*/*.png'));
    %     end