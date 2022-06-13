function [] = compare_gui_click(conf)

global username
global results
global ncount
global screen_width screen_height
% global conf

format compact;

%-------------------------------------------------------------%
%                          figure                             %
%-------------------------------------------------------------%

set(0,'units','pixels');
f = figure('Visible','off','Position',[0,0,screen_width,screen_height]);
set(f,'NumberTitle','off');
set(f,'MenuBar','none');
set(f,'Color',[0.8,0.8,0.8]);
set(f,'Close',@ky_CloseRequestFcn);
movegui(f,'center')


% --------            Instructions                --------- %
% text_word = {'键盘向左箭头: 高质量图(pass),显示下一张';
%     '键盘向右箭头: 低质量图(delete),显示下一张';
%     '-----------------------';
%     '低质量图: 1) 噪声大; 2)部分模糊/背景虚化;';
%     '3)宗教/血腥/暴力/战争; 4)有水印/@人名等文字';
%     '-----------------------';
%     '第二遍检查重点看是否有漏删的图片!';
%     '-----------------------';
%     '建议在大屏幕上全屏显示';
%     };

text_word = {'Left keyboard arrow: high-quality (pass)';
    'Right keyboard arrow: low-quality (delete)';
    '-----------------------';
    'Low quality images: ';
    '1) noisy; ';
    '2) blur;';
    '3) bloody scene/war/violence; ';
    '4) watermarks.';
    '-----------------------';
    'Full screen show';
    };
uicontrol('Style','text','String',text_word,'FontSize',14,...
    'Position',[10 300 200 300],'BackgroundColor',[0.9 0.9 0.9],'horizontalalignment','left');


% set(f,'WindowButtonDownFcn',@ButtonDownFcn);
set(f,'KeyPressFcn',@KeyPressFcn);



%-------------------------------------------------------------%
%                     Load First Pair                         %
%-------------------------------------------------------------%

image1 = imread(fullfile(conf(ncount).folder,conf(ncount).name));
imshow(image1);
set(f,'Visible','on');

%-------------------------------------------------------------%
%                    callback functions                       %
%-------------------------------------------------------------%

    function KeyPressFcn(src,event)
        %         pt = get(f,'CurrentPoint');    %获取当前点坐标
        %         x = pt(1,1);
        %         y = pt(1,2);
        %    global conf

        if strcmp(event.Key, 'rightarrow')
            disp(['键盘向右箭头删除当前;显示下一张 当前:',num2str(ncount)])
            conf(ncount).is_deleted = 1;
            conf(ncount).time_operation = datestr(now);
            delete(fullfile(conf(ncount).folder,conf(ncount).name))
            ncount = ncount+1;
            image1 = imread(fullfile(conf(ncount).folder,conf(ncount).name));
            imshow(image1);
            title(ncount);

        elseif strcmp(event.Key, 'leftarrow')
            disp(['键盘向左箭头跳过;显示下一张 当前:',num2str(ncount)])
            conf(ncount).is_deleted = 0;
            conf(ncount).time_operation = datestr(now);
            ncount = ncount+1;
            image1 = imread(fullfile(conf(ncount).folder,conf(ncount).name));
            imshow(image1);
            title(ncount);
        end
    end

    function ButtonDownFcn(~,~)
        %         pt = get(f,'CurrentPoint');    %获取当前点坐标
        %         x = pt(1,1);
        %         y = pt(1,2);
        %    global conf

        if strcmp(get(gcf,'SelectionType'),'alt')
            disp(['右键删除当前;显示下一张 当前:',num2str(ncount)])
            conf(ncount).is_deleted = 1;
            conf(ncount).time_operation = datestr(now);
            delete(fullfile(conf(ncount).folder,conf(ncount).name))
            ncount = ncount+1;
            image1 = imread(fullfile(conf(ncount).folder,conf(ncount).name));
            imshow(image1);
            title(ncount);

        elseif strcmp(get(gcf,'SelectionType'),'normal')
            disp(['左键跳过;显示下一张 当前:',num2str(ncount)])
            conf(ncount).is_deleted = 0;
            conf(ncount).time_operation = datestr(now);
            ncount = ncount+1;
            image1 = imread(fullfile(conf(ncount).folder,conf(ncount).name));
            imshow(image1);
            title(ncount);
        end
    end
end


function ky_CloseRequestFcn(~,~)
global ncount
global results
global username
global conf
button = questdlg('Ready to Quit ?', ...
    '','Yes','No','No');
switch button
    case 'Yes',
        disp('Save and Exiting ...');
        conf(ncount).is_deleted = 2;
        save(fullfile(results,['Subject#_',username,'_double_check.mat']),'conf');

        delete(gcf);
    case 'No',
        quit cancel;
end

end
