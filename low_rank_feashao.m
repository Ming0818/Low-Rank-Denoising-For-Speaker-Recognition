
clc
clear
%s='E:\NIST2003_SREE-huang\train\smg'; 
%s='E:\weiwei\shaomingguang\mymelcepttest';
%s='E:\NIST2003_SREE-huang\test\warp_window3s';
%s='E:\科研资料\语料库集\NIST2003_SREE-huang\train\warp_window3s_norm\';
s='G:\NIST2003\NIST2003_SREE-huang\test\warp_window3s\';
%desdir='D:\我的资料\桌面\exact_alm_rpca\exact_alm_rpca\low_rank\test\';

%%%%%%%%%%%%%%%%%%  提取低秩特征 %%%%%%%%%%%%%%%%%%%%%%%%%
%function  low_rank_fea(s,desdir)

temp = s;

dirs=dir(s);
dircell=struct2cell(dirs)' ; % 结构体(struct)转换成元胞类型(cell)，转置一下是让文件名按列排列。

fnames=dircell(:,1);         % 第一A列是文件名
fnumber=size(fnames,1);      % 求取表格行数，即文件个数
perCol=200;                  %每次以多少列进行低秩
for N=3:fnumber
    filename=char(fnames(N,1));       % 将cell转换为string
    filenamesave = filename(1:4);
    
    filename=[temp,'\',filename];     %%%读入需要 低秩变换的 文件
    D=load(filename); 
   D=D';
    fprintf(' 开始特征提取！');
   [row col]=size(D); 

    num=fix(col/perCol);
    
    ALL_A_hat=[];
    ALL_E_hat=[];
   for i=1:num
   newD=D(:,(1+perCol*(i-1)):perCol*i);
 
   %[A_hat E_hat iter] = exact_alm_rpca(newD);%%%%%%%%%%%低秩变换%%%%%%%%%%%%%
   [A_hat,E_hat,numIter] = proximal_gradient_rpca(newD,0.1715);

   ALL_A_hat=[ALL_A_hat A_hat];
   ALL_E_hat=[ALL_E_hat E_hat];
   end
   if(perCol*num~=col)
   newD=D(:,(perCol*num+1):col);
 
  % [A_hat E_hat iter] = exact_alm_rpca(newD);%%%对剩余部分进行低秩变换%%%%%%%%%%%%%
    [A_hat,E_hat,numIter] = proximal_gradient_rpca(newD,0.1715);

    ALL_A_hat=[ALL_A_hat A_hat];
    ALL_E_hat=[ALL_E_hat E_hat];
   end
   fprintf('  !!!!!!!!!!!!!!!!!!!!!!!!!!!完成！！');
   %fileout=[desdir,filenamesave,'.low'];
    
%   dlmwrite(strcat('E:\科研资料\语料库集\LowRank特征\trainLowRank200_A\',filenamesave),ALL_A_hat,'delimiter', ' ','-append');
%   dlmwrite(strcat('E:\科研资料\语料库集\LowRank特征\trainLowRank200_E\',filenamesave),ALL_E_hat,'delimiter', ' ','-append');
   dlmwrite(strcat('G:\低秩特征\testLowRank200_A_pg_0.1715\',filenamesave),ALL_A_hat,'delimiter', ' ','-append');
  dlmwrite(strcat('G:\低秩特征\testLowRank200_E_pg_0.1715\',filenamesave),ALL_E_hat,'delimiter', ' ','-append');
end

clc
clear
%s='E:\NIST2003_SREE-huang\train\smg';
%s='E:\weiwei\shaomingguang\mymelcepttest';
%s='E:\NIST2003_SREE-huang\test\warp_window3s';
s='G:\NIST2003\NIST2003_SREE-huang\train\warp_window3s_norm\';
%s='E:\科研资料\语料库集\NIST2003_SREE-huang\test\warp_window3s\';
%desdir='D:\我的资料\桌面\exact_alm_rpca\exact_alm_rpca\low_rank\test\';

%%%%%%%%%%%%%%%%%%  提取低秩特征 %%%%%%%%%%%%%%%%%%%%%%%%%
%function  low_rank_fea(s,desdir)

temp = s;

dirs=dir(s);
dircell=struct2cell(dirs)' ; % 结构体(struct)转换成元胞类型(cell)，转置一下是让文件名按列排列。

fnames=dircell(:,1);         % 第一A列是文件名
fnumber=size(fnames,1);      % 求取表格行数，即文件个数
perCol=200;                  %每次以多少列进行低秩
for N=3:fnumber
    filename=char(fnames(N,1));       % 将cell转换为string
    filenamesave = filename(1:4);
    
    filename=[temp,'\',filename];     %%%读入需要 低秩变换的 文件
    D=load(filename); 
    D=D';
    
    fprintf(' 开始特征提取！');
    [row col]=size(D);
 
    
    num=fix(col/perCol);
    
    ALL_A_hat=[];
    ALL_E_hat=[];
   for i=1:num
   newD=D(:,(1+perCol*(i-1)):perCol*i);
  
   
   %[A_hat E_hat iter] = exact_alm_rpca(newD);%%%%%%%%%%%低秩变换%%%%%%%%%%%%%
   [A_hat,E_hat,numIter] = proximal_gradient_rpca(newD,0.1715);
   
   ALL_A_hat=[ALL_A_hat A_hat];
   ALL_E_hat=[ALL_E_hat E_hat];
   end
   if(perCol*num~=col)
    newD=D(:,(perCol*num+1):col);
 
  % [A_hat E_hat iter] = exact_alm_rpca(newD);%%%对剩余部分进行低秩变换%%%%%%%%%%%%%
    [A_hat,E_hat,numIter] = proximal_gradient_rpca(newD,0.1715);

    ALL_A_hat=[ALL_A_hat A_hat];
    ALL_E_hat=[ALL_E_hat E_hat];
   end
   fprintf('  !!!!!!!!!!!!!!!!!!!!!!!!!!!完成！！');
   %fileout=[desdir,filenamesave,'.low'];
    
   dlmwrite(strcat('G:\低秩特征\trainLowRank200_A_pg_0.1715\',filenamesave),ALL_A_hat,'delimiter', ' ','-append');
   dlmwrite(strcat('G:\低秩特征\trainLowRank200_E_pg_0.1715\',filenamesave),ALL_E_hat,'delimiter', ' ','-append');
%    dlmwrite(strcat('E:\科研资料\语料库集\LowRank特征\testLowRank200_A\',filenamesave),ALL_A_hat,'delimiter', ' ','-append');
%   dlmwrite(strcat('E:\科研资料\语料库集\LowRank特征\testLowRank200_E\',filenamesave),ALL_E_hat,'delimiter', ' ','-append');
end



