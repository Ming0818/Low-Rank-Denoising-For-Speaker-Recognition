clc
clear

s='G:\NIST2003\NIST2003_SREE-huang\test\warp_window3s\';
%desdir='D:\�ҵ�����\����\exact_alm_rpca\exact_alm_rpca\low_rank\test\';

%%%%%%%%%%%%%%%%%%  ��ȡ�������� %%%%%%%%%%%%%%%%%%%%%%%%%
%function  low_rank_fea(s,desdir)

%s=strcat(s,'\*.norm');
temp = s;

dirs=dir(s);
dircell=struct2cell(dirs)' ; % �ṹ��(struct)ת����Ԫ������(cell)��ת��һ�������ļ����������С�

fnames=dircell(:,1);         % ��һA�����ļ���
fnumber=size(fnames,1);      % ��ȡ������������ļ�����
for N=3:fnumber
    filename=char(fnames(N,1));       % ��cellת��Ϊstring
    filenamesave = filename(1:4);
    
    filename=[temp,'\',filename];     %%%������Ҫ ���ȱ任�� �ļ�
    D=load(filename); 
    D=D';
    fprintf(' ��ʼ������ȡ��');
    [row col]=size(D); 
   
    
    
   % [A_hat E_hat iter] = exact_alm_rpca(D);%%%%%%%%%%%���ȱ任%%%%%%%%%%%%%
   [A_hat,E_hat,numIter] = proximal_gradient_rpca(D,0.124);
%      A_hat=A_hat';
%      E_hat=E_hat';
     fprintf('  !!!!!!!!!!!!!!!!!!!!!!!!!!!��ɣ���');
    %fileout=[desdir,filenamesave,'.low'];
    
   dlmwrite(strcat('G:\��������\testAll_A_pg\',filenamesave),A_hat,'delimiter', ' ','-append');%%%%%%%%%%%������Ⱥ�����%%%%%%%%%
   dlmwrite(strcat('G:\��������\testAll_E_pg\',filenamesave),E_hat,'delimiter', ' ','-append');
end





clc
clear

s='G:\NIST2003\NIST2003_SREE-huang\train\warp_window3s_norm\';
%desdir='D:\�ҵ�����\����\exact_alm_rpca\exact_alm_rpca\low_rank\test\';

%%%%%%%%%%%%%%%%%%  ��ȡ�������� %%%%%%%%%%%%%%%%%%%%%%%%%
%function  low_rank_fea(s,desdir)

%s=strcat(s,'\*.norm');
temp = s;

dirs=dir(s);
dircell=struct2cell(dirs)' ; % �ṹ��(struct)ת����Ԫ������(cell)��ת��һ�������ļ����������С�

fnames=dircell(:,1);         % ��һA�����ļ���
fnumber=size(fnames,1);      % ��ȡ������������ļ�����
for N=3:fnumber
    filename=char(fnames(N,1));       % ��cellת��Ϊstring
    filenamesave = filename(1:4);
    
    filename=[temp,'\',filename];     %%%������Ҫ ���ȱ任�� �ļ�
    D=load(filename); 
    D=D';
    fprintf(' ��ʼ������ȡ��');
    [row col]=size(D); 
   
    
    
   % [A_hat E_hat iter] = exact_alm_rpca(D);%%%%%%%%%%%���ȱ任%%%%%%%%%%%%%
   [A_hat,E_hat,numIter] = proximal_gradient_rpca(D,0.124);
%      A_hat=A_hat';
%      E_hat=E_hat';
     fprintf('  !!!!!!!!!!!!!!!!!!!!!!!!!!!��ɣ���');
    %fileout=[desdir,filenamesave,'.low'];
    
   dlmwrite(strcat('G:\��������\trainAll_A_pg\',filenamesave),A_hat,'delimiter', ' ','-append');
   dlmwrite(strcat('G:\��������\trainAll_E_pg\',filenamesave),E_hat,'delimiter', ' ','-append');
end



