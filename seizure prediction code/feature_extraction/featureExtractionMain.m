clear all
close all

addpath(genpath(pwd))
% 
myPool = parpool(8);

N{1} = [0:200];

PAT = {'327'};
for pp = 1:numel(PAT)
pat = PAT{pp};

patient = ['FR_',pat];

tic
for n = 1
    n
number = N{n};
% folder = ['\pat_',pat,'02\'];
folder = ['\pat_',pat,'02\adm_',pat,'102\rec_',pat,'0',num2str(n-1),'102\'];
% folderOut =['\pat_',pat,'02Feature\'];
folderOut =['\pat_',pat,'02Feature\',num2str(n-1),'\'];
if exist([pwd,folderOut],'dir') == 0
    mkdir([pwd,folderOut])
end
    
filepre = [pat,'0',num2str(n-1),'102_'];
filelectrode = [folder,filepre,num2str(number(1),'%04d'),'.mat'];

     for  i = 1:numel(number)
         number(i)
         file_number = [filepre,num2str(number(i),'%04d')];
         if exist([pwd,folder,num2str(file_number),'.mat'])
            output = featureExtraction(patient,folder,folderOut,filelectrode,file_number,1);
         end
     end
     
end
end
 
toc
delete(myPool);
