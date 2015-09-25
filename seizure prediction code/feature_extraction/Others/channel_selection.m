%% select 6 channels
function [ratio,selected_C] = channel_selection(file,n_channel,patient)
%% EEG channel selection 

load([pwd,'\10_1020systemCoordinates.mat']);
load([pwd,file]); %%%%%%%%%%%
if size(elec_names,2)>1
    temp = regexp(elec_names(2:end-1),'([^ ,:]*)','tokens');
    electrodes = cat(2,temp{:});
else
    electrodes = elec_names';
end

[~,txt,~] = xlsread([pwd,'\clinical_data_seizures.xlsx'],patient);
channel = [];
for i = 2:length(txt)
    tmp = regexp(txt{i,6},'([^ ,:]*)','tokens');
    out = cat(2,tmp{:});
    channel = [channel,out];
end
ind=find(ismember(channel,'FT9'));
for i = 1:numel(ind)
    channel{ind(i)} = 'T1';
end
ind=find(ismember(channel,'FT10'));
for i = 1:numel(ind)
    channel{ind(i)} = 'T2';
end
ind=find(ismember(channel,'T8'));
for i = 1:numel(ind)
    channel{ind(i)} = 'T4';
end
ind=find(ismember(channel,'T7'));
for i = 1:numel(ind)
    channel{ind(i)} = 'T3';
end
ind=find(ismember(channel,'P8'));
for i = 1:numel(ind)
    channel{ind(i)} = 'T6';
end
ind=find(ismember(channel,'P7'));
for i = 1:numel(ind)
    channel{ind(i)} = 'T5';
end
ind=find(ismember(channel,'PO7'));
for i = 1:numel(ind)
    channel{ind(i)} = 'PO5';
end
ind=find(ismember(channel,'PO8'));
for i = 1:numel(ind)
    channel{ind(i)} = 'PO6';
end

list = intersect(Channels,unique(channel));
% list = unique(channel);
% ind=find(ismember(list,'SP1'));
% if ~isempty(ind)
%     list{ind} = [];
%     list = list(~cellfun('isempty',list)); 
% end
% ind=find(ismember(list,'SP2'));
% if ~isempty(ind)
%     list{ind} = [];
%     list = list(~cellfun('isempty',list));
% end
% ind=find(ismember(list,'RS'));
% if ~isempty(ind)
%     list{ind} = [];
%     list = list(~cellfun('isempty',list));  
% end

score = zeros(size(list));
for  i = 1:numel(list)
    score(i) = numel(find(strcmp(channel,list{i})))/numel(channel);
end

[ratio,I] = sort(score,'descend');

%% the three focal channels
selected_C = list(I(1:n_channel/2));
ratio(4:end) = [];

%% the three unfocal channels
farFocal = setdiff(intersect(Channels,electrodes),list); %returns the different elemment in the first cell
if isempty(farFocal)
    farFocal = setdiff(intersect(Channels,electrodes),selected_C);
end

for i = 1:numel(farFocal)
    distanceA(i) = 0;
    for j = 1:numel(selected_C);
        distanceA(i) = distanceA(i) + pdist([Coor(find(strcmp(Channels,farFocal(i))),:);Coor(find(strcmp(Channels,selected_C(j))),:)],'euclidean');
    end
end
[~,indx] = sort(distanceA,'descend');
selected_C = [selected_C,farFocal(indx(1:n_channel/2))];



    