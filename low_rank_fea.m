clc
clear
%s='E:\NIST2003_SREE-huang\train\smg';
s='E:\NIST2003_SREE-huang\test\shaoenfram';
%s='E:\NIST2003_SREE-huang\train\warp_window3s_norm';
%desdir='D:\�ҵ�����\����\exact_alm_rpca\exact_alm_rpca\low_rank\test\';

%%%%%%%%%%%%%%%%%%  ��ȡ�������� %%%%%%%%%%%%%%%%%%%%%%%%%
%function  low_rank_fea(s,desdir)

temp = s;
%s=strcat(s,'\*.norm');
dirs=dir(s);
dircell=struct2cell(dirs)' ; % �ṹ��(struct)ת����Ԫ������(cell)��ת��һ�������ļ����������С�

fnames=dircell(:,1);         % ��һA�����ļ���
fnumber=size(fnames,1);      % ��ȡ������������ļ�����
for N=3:fnumber
    filename=char(fnames(N,1));       % ��cellת��Ϊstring
    filenamesave = filename(1:4);
    
    filename=[temp,'\',filename];     %%%������Ҫ ���ȱ任�� �ļ�
    D=load(filename); 
    fprintf(' ��ʼ������ȡ��');
    [row col]=size(D); 
    num=fix(row/50);
    B_hat=[];
    for i=1:num
    new=D((1+50*(i-1)):50*i,:);
    [A_hat E_hat iter] = exact_alm_rpca(new);%%%%%%%%%%%���ȱ任%%%%%%%%%%%%%
   %[A_hat,E_hat,numIter] = proximal_gradient_rpca(D,0.124);
   B_hat=[B_hat;A_hat];
    end
     fprintf('  !!!!!!!!!!!!!!!!!!!!!!!!!!!��ɣ���');
    %fileout=[desdir,filenamesave,'.low'];
    
   dlmwrite(filenamesave,B_hat,'delimiter', ' ','-append');%%%%%%%%%%%������Ⱥ�����%%%%%%%%%
end

%end
