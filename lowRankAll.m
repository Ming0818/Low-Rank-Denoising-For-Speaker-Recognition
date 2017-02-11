clc
clear

s='G:\NIST2003\NIST2003_SREE-huang\test\warp_window3s\';
%desdir='D:\我的资料\桌面\exact_alm_rpca\exact_alm_rpca\low_rank\test\';

%%%%%%%%%%%%%%%%%%  提取低秩特征 %%%%%%%%%%%%%%%%%%%%%%%%%
%function  low_rank_fea(s,desdir)

%s=strcat(s,'\*.norm');
temp = s;

dirs=dir(s);
dircell=struct2cell(dirs)' ; % 结构体(struct)转换成元胞类型(cell)，转置一下是让文件名按列排列。

fnames=dircell(:,1);         % 第一A列是文件名
fnumber=size(fnames,1);      % 求取表格行数，即文件个数
for N=3:fnumber
    filename=char(fnames(N,1));       % 将cell转换为string
    filenamesave = filename(1:4);
    
    filename=[temp,'\',filename];     %%%读入需要 低秩变换的 文件
    D=load(filename); 
    D=D';
    fprintf(' 开始特征提取！');
    [row col]=size(D); 
   
    
    
   % [A_hat E_hat iter] = exact_alm_rpca(D);%%%%%%%%%%%低秩变换%%%%%%%%%%%%%
   [A_hat,E_hat,numIter] = proximal_gradient_rpca(D,0.124);
%      A_hat=A_hat';
%      E_hat=E_hat';
     fprintf('  !!!!!!!!!!!!!!!!!!!!!!!!!!!完成！！');
    %fileout=[desdir,filenamesave,'.low'];
    
   dlmwrite(strcat('G:\低秩特征\testAll_A_pg\',filenamesave),A_hat,'delimiter', ' ','-append');%%%%%%%%%%%保存低秩后数据%%%%%%%%%
   dlmwrite(strcat('G:\低秩特征\testAll_E_pg\',filenamesave),E_hat,'delimiter', ' ','-append');
end





clc
clear

s='G:\NIST2003\NIST2003_SREE-huang\train\warp_window3s_norm\';
%desdir='D:\我的资料\桌面\exact_alm_rpca\exact_alm_rpca\low_rank\test\';

%%%%%%%%%%%%%%%%%%  提取低秩特征 %%%%%%%%%%%%%%%%%%%%%%%%%
%function  low_rank_fea(s,desdir)

%s=strcat(s,'\*.norm');
temp = s;

dirs=dir(s);
dircell=struct2cell(dirs)' ; % 结构体(struct)转换成元胞类型(cell)，转置一下是让文件名按列排列。

fnames=dircell(:,1);         % 第一A列是文件名
fnumber=size(fnames,1);      % 求取表格行数，即文件个数
for N=3:fnumber
    filename=char(fnames(N,1));       % 将cell转换为string
    filenamesave = filename(1:4);
    
    filename=[temp,'\',filename];     %%%读入需要 低秩变换的 文件
    D=load(filename); 
    D=D';
    fprintf(' 开始特征提取！');
    [row col]=size(D); 
   
    
    
   % [A_hat E_hat iter] = exact_alm_rpca(D);%%%%%%%%%%%低秩变换%%%%%%%%%%%%%
   [A_hat,E_hat,numIter] = proximal_gradient_rpca(D,0.124);
%      A_hat=A_hat';
%      E_hat=E_hat';
     fprintf('  !!!!!!!!!!!!!!!!!!!!!!!!!!!完成！！');
    %fileout=[desdir,filenamesave,'.low'];
    
   dlmwrite(strcat('G:\低秩特征\trainAll_A_pg\',filenamesave),A_hat,'delimiter', ' ','-append');
   dlmwrite(strcat('G:\低秩特征\trainAll_E_pg\',filenamesave),E_hat,'delimiter', ' ','-append');
end



