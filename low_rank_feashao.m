
clc
clear
%s='E:\NIST2003_SREE-huang\train\smg'; 
%s='E:\weiwei\shaomingguang\mymelcepttest';
%s='E:\NIST2003_SREE-huang\test\warp_window3s';
%s='E:\��������\���Ͽ⼯\NIST2003_SREE-huang\train\warp_window3s_norm\';
s='G:\NIST2003\NIST2003_SREE-huang\test\warp_window3s\';
%desdir='D:\�ҵ�����\����\exact_alm_rpca\exact_alm_rpca\low_rank\test\';

%%%%%%%%%%%%%%%%%%  ��ȡ�������� %%%%%%%%%%%%%%%%%%%%%%%%%
%function  low_rank_fea(s,desdir)

temp = s;

dirs=dir(s);
dircell=struct2cell(dirs)' ; % �ṹ��(struct)ת����Ԫ������(cell)��ת��һ�������ļ����������С�

fnames=dircell(:,1);         % ��һA�����ļ���
fnumber=size(fnames,1);      % ��ȡ������������ļ�����
perCol=200;                  %ÿ���Զ����н��е���
for N=3:fnumber
    filename=char(fnames(N,1));       % ��cellת��Ϊstring
    filenamesave = filename(1:4);
    
    filename=[temp,'\',filename];     %%%������Ҫ ���ȱ任�� �ļ�
    D=load(filename); 
   D=D';
    fprintf(' ��ʼ������ȡ��');
   [row col]=size(D); 

    num=fix(col/perCol);
    
    ALL_A_hat=[];
    ALL_E_hat=[];
   for i=1:num
   newD=D(:,(1+perCol*(i-1)):perCol*i);
 
   %[A_hat E_hat iter] = exact_alm_rpca(newD);%%%%%%%%%%%���ȱ任%%%%%%%%%%%%%
   [A_hat,E_hat,numIter] = proximal_gradient_rpca(newD,0.1715);

   ALL_A_hat=[ALL_A_hat A_hat];
   ALL_E_hat=[ALL_E_hat E_hat];
   end
   if(perCol*num~=col)
   newD=D(:,(perCol*num+1):col);
 
  % [A_hat E_hat iter] = exact_alm_rpca(newD);%%%��ʣ�ಿ�ֽ��е��ȱ任%%%%%%%%%%%%%
    [A_hat,E_hat,numIter] = proximal_gradient_rpca(newD,0.1715);

    ALL_A_hat=[ALL_A_hat A_hat];
    ALL_E_hat=[ALL_E_hat E_hat];
   end
   fprintf('  !!!!!!!!!!!!!!!!!!!!!!!!!!!��ɣ���');
   %fileout=[desdir,filenamesave,'.low'];
    
%   dlmwrite(strcat('E:\��������\���Ͽ⼯\LowRank����\trainLowRank200_A\',filenamesave),ALL_A_hat,'delimiter', ' ','-append');
%   dlmwrite(strcat('E:\��������\���Ͽ⼯\LowRank����\trainLowRank200_E\',filenamesave),ALL_E_hat,'delimiter', ' ','-append');
   dlmwrite(strcat('G:\��������\testLowRank200_A_pg_0.1715\',filenamesave),ALL_A_hat,'delimiter', ' ','-append');
  dlmwrite(strcat('G:\��������\testLowRank200_E_pg_0.1715\',filenamesave),ALL_E_hat,'delimiter', ' ','-append');
end

clc
clear
%s='E:\NIST2003_SREE-huang\train\smg';
%s='E:\weiwei\shaomingguang\mymelcepttest';
%s='E:\NIST2003_SREE-huang\test\warp_window3s';
s='G:\NIST2003\NIST2003_SREE-huang\train\warp_window3s_norm\';
%s='E:\��������\���Ͽ⼯\NIST2003_SREE-huang\test\warp_window3s\';
%desdir='D:\�ҵ�����\����\exact_alm_rpca\exact_alm_rpca\low_rank\test\';

%%%%%%%%%%%%%%%%%%  ��ȡ�������� %%%%%%%%%%%%%%%%%%%%%%%%%
%function  low_rank_fea(s,desdir)

temp = s;

dirs=dir(s);
dircell=struct2cell(dirs)' ; % �ṹ��(struct)ת����Ԫ������(cell)��ת��һ�������ļ����������С�

fnames=dircell(:,1);         % ��һA�����ļ���
fnumber=size(fnames,1);      % ��ȡ������������ļ�����
perCol=200;                  %ÿ���Զ����н��е���
for N=3:fnumber
    filename=char(fnames(N,1));       % ��cellת��Ϊstring
    filenamesave = filename(1:4);
    
    filename=[temp,'\',filename];     %%%������Ҫ ���ȱ任�� �ļ�
    D=load(filename); 
    D=D';
    
    fprintf(' ��ʼ������ȡ��');
    [row col]=size(D);
 
    
    num=fix(col/perCol);
    
    ALL_A_hat=[];
    ALL_E_hat=[];
   for i=1:num
   newD=D(:,(1+perCol*(i-1)):perCol*i);
  
   
   %[A_hat E_hat iter] = exact_alm_rpca(newD);%%%%%%%%%%%���ȱ任%%%%%%%%%%%%%
   [A_hat,E_hat,numIter] = proximal_gradient_rpca(newD,0.1715);
   
   ALL_A_hat=[ALL_A_hat A_hat];
   ALL_E_hat=[ALL_E_hat E_hat];
   end
   if(perCol*num~=col)
    newD=D(:,(perCol*num+1):col);
 
  % [A_hat E_hat iter] = exact_alm_rpca(newD);%%%��ʣ�ಿ�ֽ��е��ȱ任%%%%%%%%%%%%%
    [A_hat,E_hat,numIter] = proximal_gradient_rpca(newD,0.1715);

    ALL_A_hat=[ALL_A_hat A_hat];
    ALL_E_hat=[ALL_E_hat E_hat];
   end
   fprintf('  !!!!!!!!!!!!!!!!!!!!!!!!!!!��ɣ���');
   %fileout=[desdir,filenamesave,'.low'];
    
   dlmwrite(strcat('G:\��������\trainLowRank200_A_pg_0.1715\',filenamesave),ALL_A_hat,'delimiter', ' ','-append');
   dlmwrite(strcat('G:\��������\trainLowRank200_E_pg_0.1715\',filenamesave),ALL_E_hat,'delimiter', ' ','-append');
%    dlmwrite(strcat('E:\��������\���Ͽ⼯\LowRank����\testLowRank200_A\',filenamesave),ALL_A_hat,'delimiter', ' ','-append');
%   dlmwrite(strcat('E:\��������\���Ͽ⼯\LowRank����\testLowRank200_E\',filenamesave),ALL_E_hat,'delimiter', ' ','-append');
end



