function [aVLF,aLF,aHF,aTotal,pVLF,pLF,pHF,nLF,nHF,LFHF,peakVLF,peakLF,peakHF]= timeFreqHRV(ibi,VLF,LF,HF,fs)
%timeFreqHRV - calculates time-freq HRV using ar, lomb, and CWT methods
%
% Inputs:   ibi = 2Dim array of [time (s) ,inter-beat interval (s)]
%           VLF,LF,HF = arrays containing limits of VLF, LF, and HF 
%                       freq. bands
%           fs = cubic spline interpolation rate / resample rate (Hz)
% Outputs:  output is a structure containg all HRV. One field for each
%               PSD method
%           Output units include:
%               peakHF,peakLF,peakVLF (Hz)
%               aHF,aLF,aVLF (ms^2)
%               pHF,pLF,pVLF (%)
%               nHF,nLF,nVLF (%)
%               lfhf
% Usage:   n/a     
    
    t=ibi(:,1); %time
    y=ibi(:,2);        
    clear ibi; %don't need it anymore
    
    maxF=fs/2;
    lomb_t =t;
    nfft = 2*fs;
    
    if numel(y) == 1
        aVLF = nan; aLF = nan; aHF = nan; aTotal = nan; pVLF = nan; pLF = nan; pHF = nan; ...
            nLF = nan; nHF = nan; LFHF = nan; peakVLF = nan; peakLF = nan; peakHF = nan;
    else
        [lomb_psd,lomb_f] = plomb(y,t/1000);
        nan_indx = find(isnan(lomb_psd));
        lomb_psd(nan_indx) = [];
        lomb_f(nan_indx) = [];
    %     [lomb_psd,lomb_f]= ...
    %             calcLomb(t,y,nfft,maxF);
        [aVLF,aLF,aHF,aTotal,pVLF,pLF,pHF,nLF,nHF,LFHF,peakVLF,peakLF,peakHF]=calcHRV(lomb_f,lomb_psd,VLF,LF,HF);
    end
  
end
% 
% function [PSD,F]=calcLomb(t,y,nfft,maxF)
% %calLomb - Calculates PSD using windowed Lomb-Scargle method.   
%     
%     deltaF=maxF/nfft;        
%     F = linspace(0.0,maxF-deltaF,nfft)';
%     
%         %remove linear trend
%         y=detrend(y,'linear');
%         
%         y=y-mean(y); %remove mean
%         
%         %Calculate un-normalized lomb PSD        
%         PSD(:,1)=lomb2(y,t,F,false); 
%         
% end

function [aVLF,aLF,aHF,aTotal,pVLF,pLF,pHF,nLF,nHF,LFHF,peakVLF,peakLF,peakHF]=calcHRV(F,PSD,VLF,LF,HF)
% calcAreas - Calulates areas/energy under the PSD curve within the freq
% bands defined by VLF, LF, and HF. Returns areas/energies as ms^2,
% percentage, and normalized units. Also returns LF/HF ratio.
%
% Inputs:
%   PSD: PSD vector
%   F: Freq vector
%   VLF, LF, HF: array containing VLF, LF, and HF freq limits
%   flagVLF: flag to decide whether to calculate VLF hrv
% Output:
%
% Usage:
%   
%
% Ref: Modified from Gary Clifford's ECG Toolbox: calc_lfhf.m   
%     
%     nPSD=size(PSD,2);
%     z=zeros(nPSD,1);
%     output= struct('aVLF',z, 'aLF',z, 'aHF',z, 'aTotal',z, 'pVLF',z, ...
%         'pLF',z, 'pHF',z, 'nLF',z, 'nHF',z, 'LFHF',z, 'rLFHF',0, ...
%         'peakVLF',z, 'peakLF',z, 'peakHF',z);
%         
        a=calcAreas(F,PSD(:,1),VLF,LF,HF);

        %create output structure
        aVLF=a.aVLF;
        aLF=a.aLF;
        aHF=a.aHF;
        aTotal=a.aTotal;
        pVLF=a.pVLF;
        pLF=a.pLF;
        pHF=a.pHF;
        nLF=a.nLF;
        nHF=a.nHF;
        LFHF=a.LFHF;
        peakVLF=a.peakVLF;
        peakLF=a.peakLF;
        peakHF=a.peakHF;
end

function output=calcAreas(F,PSD,VLF,LF,HF,flagNorm)
%calcAreas - Calulates areas/energy under the PSD curve within the freq
%bands defined by VLF, LF, and HF. Returns areas/energies as ms^2,
%percentage, and normalized units. Also returns LF/HF ratio.
%
%Inputs:
%   PSD: PSD vector
%   F: Freq vector
%   VLF, LF, HF: array containing VLF, LF, and HF freq limits
%   flagNormalize: option to normalize PSD to max(PSD)
%Output:
%
%Usage:
%   
%
% Reference: This code is based on the calc_lfhf.m function from Gary
% Clifford's ECG Toolbox.    

    if nargin<6
       flagNorm=false;
    end
    
    %normalize PSD if needed
    if flagNorm
        PSD=PSD/max(PSD);
    end

    % find the indexes corresponding to the VLF, LF, and HF bands
    iVLF= (F>=VLF(1)) & (F<=VLF(2));
    iLF = (F>=LF(1)) & (F<=LF(2));
    iHF = (F>=HF(1)) & (F<=HF(2));
      
    %Find peaks
      %VLF Peak
      tmpF=F(iVLF);
      tmppsd=PSD(iVLF);
      [pks,ipks] = zipeaks(tmppsd);
      if ~isempty(pks)
        [tmpMax i]=max(pks);        
        peakVLF=tmpF(ipks(i));
      else
        [tmpMax i]=max(tmppsd);
        peakVLF=tmpF(i);
      end
      %LF Peak
      tmpF=F(iLF);
      tmppsd=PSD(iLF);
      [pks,ipks] = zipeaks(tmppsd);
      if ~isempty(pks)
        [tmpMax i]=max(pks);
        peakLF=tmpF(ipks(i));
      else
        [tmpMax i]=max(tmppsd);
        peakLF=tmpF(i);
      end
      %HF Peak
      tmpF=F(iHF);
      tmppsd=PSD(iHF);
      [pks,ipks] = zipeaks(tmppsd);
      if ~isempty(pks)
        [tmpMax i]=max(pks);        
        peakHF=tmpF(ipks(i));
      else
        [tmpMax i]=max(tmppsd);
        peakHF=tmpF(i);
      end 
      
    % calculate raw areas (power under curve), within the freq bands
    aVLF=trapz(F(iVLF),PSD(iVLF),1);
    aLF=trapz(F(iLF),PSD(iLF),1);
    aHF=trapz(F(iHF),PSD(iHF),1);
    aTotal=aVLF+aLF+aHF;
        
    %calculate areas relative to the total area (%)
    pVLF=(aVLF/aTotal)*100;
    pLF=(aLF/aTotal)*100;
    pHF=(aHF/aTotal)*100;
    
    %calculate normalized areas (relative to HF+LF, n.u.)
    nLF=aLF/(aLF+aHF);
    nHF=aHF/(aLF+aHF);
    
    %calculate LF/HF ratio
    lfhf =aLF/aHF;
            
     %create output structure
    output.aVLF=aVLF; % round
    output.aLF=aLF;
    output.aHF=aHF;
    output.aTotal=aTotal;
    
    output.pVLF=pVLF;
    output.pLF=pLF;
    output.pHF=pHF;
    
    output.nLF=nLF;
    output.nHF=nHF;
    output.LFHF=lfhf;
    if ~isempty(peakVLF)
        output.peakVLF=peakVLF(1);
    else
        output.peakVLF=0;
    end
    if ~isempty(peakLF)
        output.peakLF=peakLF(1);
    else
        output.peakLF=0;
    end
    if ~isempty(peakHF)
        output.peakHF=peakHF(1);
    else
        output.peakHF=0;
    end
end

function [pks locs]=zipeaks(y)
%zippeaks: finds local maxima of input signal y
%Usage:  peak=zipeaks(y);
%Returns 2x(number of maxima) array
%pks = value at maximum
%locs = index value for maximum
%
%Reference:  2009, George Zipfel (Mathworks File Exchange #24797)

%check dimentions
if isempty(y)
%     Warning('Empty input array')
    pks=[]; locs=[];
    return
end
[rows cols] = size(y);
if cols==1 && rows>1 %all data in 1st col
    y=y';
elseif cols==1 && rows==1 
%     Warning('Short input array')
    pks=[]; locs=[];
    return    
end         
    
%Find locations of local maxima
%yD=1 at maxima, yD=0 otherwise, end point maxima excluded
    N=length(y)-2;
    yD=[0 (sign(sign(y(2:N+1)-y(3:N+2))-sign(y(1:N)-y(2:N+1))-.1)+1) 0];
%Indices of maxima and corresponding values of y
    Y=logical(yD);
    I=1:length(Y);
    locs=I(Y);
    pks=y(Y);
end
