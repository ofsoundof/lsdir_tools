

addpath(genpath('./.'));


global username
global imgDataPath
global results
global screen_width screen_height

%%==================Needs to be modified====================%%
%%%

imgDataPath = '~/projects/dataset/Original/train'; % set the path
username = 'yawei'; % set the user name

%%==============================================%%


results   =  'results';
screen_width = 1000;
screen_height = 600;




























































































if ~exist(fullfile(results,['Subject#_',username,'_double_check.mat']),'file')
    registration(username);
    load(fullfile(results,['Subject#_',username,'_double_check.mat']),'conf');
else
    load(fullfile(results,['Subject#_',username,'_double_check.mat']),'conf');
end

global ncount
jj = [];
for j = 1:length(conf)
    if conf(j).is_deleted == 2
        jj = [jj,j];
    end
end
ncount = max(jj);

%自己设置ncount

disp(['======从第',num2str(ncount),'个开始======'])
%conf.time_start{end+1} = datestr(now);
compare_gui_click(conf);

rmpath(genpath('./.'))
