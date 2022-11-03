function [model,fval]=MCD_DA(K_mat, y, C, tol,c_num)
% the multicircle disassembly based decomposition algorithm
%
% USAGE: model=Chinese_ring_algorithm(K_mat, y, C, tol,c_num)
%
% INPUT:
%
%   K_mat: c_num x n , c_num x n kernel matrix, and n denotes the number of
%   samples
%   y: 1 x n vector of labels, -1 or 1
%   C: a regularization parameter such that 0 <= alpha_i <= C
%   tol: 4 x tolerance for terminating criterion [tol1,tol2,tol3,tol4]
%   c_num: the number of classes
%
% OUTPUT:
%
%   model.alpha: c_num x n lagrange multiplier coefficient
%   fval: the values of objective function for each iteration
% Designed by Gao Tong, 2021
global P_SVM;
global nested_rectangular_circle;
global Z_circle;
i_n=[1,1,1,1];
ntp = size(K_mat,1);
alpha_i_yi=zeros(c_num,ntp/c_num);
for i=1:c_num
    temp=zeros(1,ntp/c_num);
    temp(y==i)=1;
    alpha_i_yi(i,:)=temp;
end
alpha_i_yi=alpha_i_yi(:)';
ntpT=cell(c_num,c_num);
for i=1:c_num
    for j=1:c_num
       ntpT{i,j}=zeros(1,ntp);
       temp=zeros(1,ntp/c_num);
       temp(y==i)=1;
       temp_vec1=kron(temp,ones(1,c_num));
       temp=zeros(1,c_num);
       temp(j)=1;
       temp_vec2=kron(ones(1,ntp/c_num),temp);
       ntpT{i,j}(temp_vec1==1&temp_vec2==1)=1;
    end
end
alpha_num=size(K_mat,1);
alpha_init = 0*ones(1,alpha_num);
%Initialize the variables
P_SVM.alpha_i_yi=alpha_i_yi;
P_SVM.ntpT=ntpT;
P_SVM.y = y;
P_SVM.C = C;
P_SVM.alpha = alpha_init;
P_SVM.tol=tol;
P_SVM.ntp = ntp; %number of training points
P_SVM.epsilon = 10^(-4);
P_SVM.Kcache = K_mat; %kernel evaluations
temp=2*P_SVM.Kcache*(P_SVM.alpha')-2*ones(size(P_SVM.alpha,2),1);
P_SVM.grad=temp;
P_SVM.b=0;
P_SVM.p=0;
object_F=@(alpha_v) alpha_v*K_mat*alpha_v'-2*alpha_v*ones(size(alpha_v,2),1);
loop_ul{1}=[1,0,1,0;0,1,0,1];
loop_ul{2}=[0,1,0,1;1,0,1,0];
iter_convergence_temp=ones(size(ntpT,1),size(ntpT,2));%ones(size(ntpT,1),size(ntpT,2));
for i=1:size(ntpT,1)
    iter_convergence_temp(i,i)=0;
end
iter=0;
% disp('iter fval_current');
while true
    alpha_buffer=P_SVM.alpha;
%loop disassembly
iter_convergence1=iter_convergence_temp;
iter1=0;
while ~isempty(find(iter_convergence1==1))
iter_convergence1=iter_convergence_temp;
    for i=1:size(ntpT,1)
        for j=1:size(ntpT,2)
        var_increase= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{i,j}==1&P_SVM.alpha_i_yi==0);
        var_decrease= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{i,j}==1&P_SVM.alpha_i_yi==0);
        grad_decrease=P_SVM.grad(var_decrease);
        grad_increase=P_SVM.grad(var_increase);
        grad_sort=grad_increase*ones(1,size(grad_decrease,1))-ones(size(grad_increase,1),1)*grad_decrease';
        while true
            [B,I]=min(grad_sort(:));
            if isempty(B)||B>=-P_SVM.tol(1)
                iter_convergence1(i,j)=0;
                step_value=0;
                break;
            end
            [I1, I2] = ind2sub(size(grad_sort),I);
            step_grad_order1=B;
            deta_up=P_SVM.C-P_SVM.alpha(var_increase(I1));
            deta_low=P_SVM.alpha(var_decrease(I2));
            distance=min([deta_up,deta_low]);
            step_grad_order2=2*[1,-1]*P_SVM.Kcache([var_increase(I1),var_decrease(I2)],[var_increase(I1),var_decrease(I2)])*[1;-1];
            step_optimal_value=-step_grad_order1/(step_grad_order2+eps);
            step_value=min([step_optimal_value,distance]);
                break;
        end
        if step_value>0
                iter1=iter1+1;
            P_SVM.alpha(var_increase(I1))=P_SVM.alpha(var_increase(I1))+step_value;
            P_SVM.alpha(var_decrease(I2))=P_SVM.alpha(var_decrease(I2))-step_value;
            P_SVM.grad=P_SVM.grad+2*P_SVM.Kcache(:,[var_increase(I1),var_decrease(I2)])*[step_value;-step_value];%updata gradient of object function
        end
        end
    end
    if iter1>i_n(1)
%          disp('circle');
%         object_F(P_SVM.alpha)
        break;
    end
end
% rectanglar-circle disassembly
if c_num<=3
 loop_convergence2=1;%1
else
loop_convergence2=0;
end
iter2=1;
while loop_convergence2==0
    loop_convergence_one2=0;
        for i=1:c_num-1
            if loop_convergence_one2==1
                break;
            end
            for j=i+1:c_num
                if loop_convergence_one2==1
                    break;
                end
                for k=1:c_num-1
                    if loop_convergence_one2==1
                        break;
                    end
                    for l=k+1:c_num
                        if loop_convergence_one2==1
                            break;
                        end
                        if i==k||l==j||i==l||j==k
                            continue;
                        end
                        var_increase1= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{i,k}==1&P_SVM.alpha_i_yi==0);
                        var_decrease1= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{i,k}==1&P_SVM.alpha_i_yi==0);
                        var_increase2= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{j,k}==1&P_SVM.alpha_i_yi==0);
                        var_decrease2= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{j,k}==1&P_SVM.alpha_i_yi==0);
                        var_increase3= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{j,l}==1&P_SVM.alpha_i_yi==0);
                        var_decrease3= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{j,l}==1&P_SVM.alpha_i_yi==0);
                        var_increase4= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{i,l}==1&P_SVM.alpha_i_yi==0);
                        var_decrease4= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{i,l}==1&P_SVM.alpha_i_yi==0);
                        grad_d=zeros(2,4);
                        grad_I=zeros(2,4);
                        if (isempty(var_decrease1)||isempty(var_increase2)||isempty(var_decrease3)||isempty(var_increase4))&&...
                                (isempty(var_increase1)||isempty(var_decrease2)||isempty(var_increase3)||isempty(var_decrease4))
                            continue;
                        end
                        if ~(isempty(var_decrease1)||isempty(var_increase2)||isempty(var_decrease3)||isempty(var_increase4))
                            [temp_B,temp_I]=max(P_SVM.grad(var_decrease1));
                            grad_d(1,1)=-temp_B;
                            grad_I(1,1)=var_decrease1(temp_I);
                            [temp_B,temp_I]=min(P_SVM.grad(var_increase2));
                            grad_d(1,2)=temp_B;
                            grad_I(1,2)=var_increase2(temp_I);
                            [temp_B,temp_I]=max(P_SVM.grad(var_decrease3));
                            grad_d(1,3)=-temp_B;
                            grad_I(1,3)=var_decrease3(temp_I);
                            [temp_B,temp_I]=min(P_SVM.grad(var_increase4));
                            grad_d(1,4)=temp_B;
                            grad_I(1,4)=var_increase4(temp_I);
                        end
                        if ~(isempty(var_increase1)||isempty(var_decrease2)||isempty(var_increase3)||isempty(var_decrease4))
                            [temp_B,temp_I]=min(P_SVM.grad(var_increase1));
                            grad_d(2,1)=temp_B;
                            grad_I(2,1)=var_increase1(temp_I);
                            [temp_B,temp_I]=max(P_SVM.grad(var_decrease2));
                            grad_d(2,2)=-temp_B;
                            grad_I(2,2)=var_decrease2(temp_I);
                            [temp_B,temp_I]=min(P_SVM.grad(var_increase3));
                            grad_d(2,3)=temp_B;
                            grad_I(2,3)=var_increase3(temp_I);
                            [temp_B,temp_I]=max(P_SVM.grad(var_decrease4));
                            grad_d(2,4)=-temp_B;
                            grad_I(2,4)=var_decrease4(temp_I);
                        end
                        grad_loop=grad_d*ones(4,1);
                        temp_lr=find(grad_loop<-P_SVM.tol(2));
                        if ~isempty(temp_lr)
                            loop_I=grad_I(temp_lr(1),:);
                            deta_up=P_SVM.C-P_SVM.alpha(loop_I(loop_ul{temp_lr(1)}(2,:)==1));
                            deta_low=P_SVM.alpha(loop_I(loop_ul{temp_lr(1)}(1,:)==1));
                            distance=min([deta_up,deta_low]);
                            step_optimal_value=-grad_d(temp_lr(1),:)*ones(4,1)/(2*[1,1,-1,-1]*P_SVM.Kcache([loop_I(loop_ul{temp_lr(1)}(2,:)==1),loop_I(loop_ul{temp_lr(1)}(1,:)==1)],[loop_I(loop_ul{temp_lr(1)}(2,:)==1),loop_I(loop_ul{temp_lr(1)}(1,:)==1)])*[1,1,-1,-1]');
                                step_value=min([step_optimal_value,distance]);
                                P_SVM.alpha(loop_I(loop_ul{temp_lr(1)}(2,:)==1))=P_SVM.alpha(loop_I(loop_ul{temp_lr(1)}(2,:)==1))+step_value;
                                P_SVM.alpha(loop_I(loop_ul{temp_lr(1)}(1,:)==1))=P_SVM.alpha(loop_I(loop_ul{temp_lr(1)}(1,:)==1))-step_value;
                                P_SVM.grad=P_SVM.grad+2*P_SVM.Kcache(:,[loop_I(loop_ul{temp_lr(1)}(2,:)==1),loop_I(loop_ul{temp_lr(1)}(1,:)==1)])*[step_value;step_value;-step_value;-step_value];%updata gradient of object function
                                loop_convergence_one2=1;
                        end
                    end
                end
            end
        end
        if loop_convergence_one2==0||iter2>i_n(2)
           loop_convergence2=1;
        end
        iter2=iter2+1;
end
%nested rectangular circle disassembly
if c_num>=3%1
% examize the NULL vertex
NULL_v_increase=diag(ones(1,c_num));
NULL_v_decrease=diag(ones(1,c_num));
for i=1:c_num
    for j=1:c_num
        temp_ind=find(P_SVM.alpha(P_SVM.ntpT{i,j}==1)<P_SVM.C-P_SVM.epsilon, 1);
        if isempty(temp_ind)
            NULL_v_increase(i,j)=1;
        end
        temp_ind=find(P_SVM.alpha(P_SVM.ntpT{i,j}==1)>P_SVM.epsilon, 1);
        if isempty(temp_ind)
            NULL_v_decrease(i,j)=1;
        end
    end
end
loop_convergence3=0;
iter3=0;
    while loop_convergence3==0
    iter3=iter3+1;
        [NULL_increase_ind_1,NULL_increase_ind_2]=find(NULL_v_increase==1);
        [NULL_decrease_ind_1,NULL_decrease_ind_2]=find(NULL_v_decrease==1);
        %search NULL_v_increase
        convergence_I=ones(1,size(NULL_increase_ind_1,1));
        convergence_D=ones(1,size(NULL_decrease_ind_1,1));
        for g=1:size(NULL_increase_ind_1,1)
            nested_rectangular_circle.grad=[];
            nested_rectangular_circle.v=[];
            nested_rectangular_circle.ind=[];
            nested_rectangular_circle.step=C;
            nested_rectangular_circle.v_c=[];        
            i1=NULL_increase_ind_1(g);
            k1=NULL_increase_ind_2(g);
            nested_rectangular_circle.v_c=[nested_rectangular_circle.v_c,[i1;k1]];        
            iter_multiple_loop_i=1;
            [j2,l2]=search_first_vertex(1,c_num,i1,k1,1);
            if j2~=0
            while true  

               [j2,l2]=search_first_vertex(1,c_num,i1,k1,(-1)^iter_multiple_loop_i);
            if j2~=0
                    multiple_loop_update();
                    break;
            else
                [j2,l2]=search_vertex(1,c_num,i1,k1,(-1)^iter_multiple_loop_i);
                if j2~=0
                        i1=j2;
                        k1=l2;
                else
                    break;
                end
            end
            iter_multiple_loop_i=iter_multiple_loop_i+1;
            if iter_multiple_loop_i>=c_num
                j2=0;
                break;
            end
            end
            end
            convergence_I(g)=j2;
        end
        %search NULL_v_decrease
        for g=1:size(NULL_decrease_ind_1,1)
            nested_rectangular_circle.grad=[];
            nested_rectangular_circle.v=[];
            nested_rectangular_circle.ind=[];
            nested_rectangular_circle.step=C;
            nested_rectangular_circle.v_c=[];        
            i1=NULL_decrease_ind_1(g);
            k1=NULL_decrease_ind_2(g);
            nested_rectangular_circle.v_c=[nested_rectangular_circle.v_c,[i1;k1]];        
            iter_multiple_loop_d=1;
            [j2_,l2_]=search_first_vertex(1,c_num,i1,k1,-1);
            if j2_~=0
            while true  

               [j2_,l2_]=search_first_vertex(1,c_num,i1,k1,(-1)^(iter_multiple_loop_d+1));
            if j2_~=0
                    multiple_loop_update();
                    break;
            else
                [j2_,l2_]=search_vertex(1,c_num,i1,k1,(-1)^(iter_multiple_loop_d+1));
                if j2_~=0
                        i1=j2_;
                        k1=l2_;
                else
                    break;
                end
            end
            iter_multiple_loop_d=iter_multiple_loop_d+1;
            if iter_multiple_loop_d>=c_num
                j2_=0;
                break;
            end
            end
            end
            convergence_D(g)=j2_;
        end
        if sum([convergence_D,convergence_I])==0||iter3>i_n(3)
            loop_convergence3=1;
        end
        iter3=iter3+1;
    end
end
%Z-circle disassembly
loop_convergence4=0;
In_V_Mat=zeros(c_num,c_num);
In_I_Mat=zeros(c_num,c_num);
De_V_Mat=zeros(c_num,c_num);
De_I_Mat=zeros(c_num,c_num);
for i=1:c_num
    for j=1:c_num
        var_increase= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{i,j}==1&P_SVM.alpha_i_yi==0);
        grad_increase=P_SVM.grad(var_increase);
        [B,I]=min(grad_increase);
        if ~isempty(B)
            In_V_Mat(i,j)=B;
            In_I_Mat(i,j)=var_increase(I);
        else
            In_V_Mat(i,j)=NaN;
            In_I_Mat(i,j)=NaN;
        end
        var_decrease= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{i,j}==1&P_SVM.alpha_i_yi==0);
        grad_decrease=P_SVM.grad(var_decrease);
        [B,I]=max(grad_decrease);
        if ~isempty(B)
            De_V_Mat(i,j)=B;
            De_I_Mat(i,j)=var_decrease(I);
        else
            De_V_Mat(i,j)=NaN;
            De_I_Mat(i,j)=NaN;
        end
    end
end
iter4=0;
while loop_convergence4==0
    results_set=zeros(1,2);
    for i=1:c_num
        for j=1:c_num
            if In_V_Mat(i,j)<-0.001
            In_V_Mat(~isnan(In_V_Mat))=P_SVM.grad(In_I_Mat(~isnan(In_V_Mat)));
            results=search_Z_loop_vertex_in(1,c_num,i,j,In_V_Mat,In_I_Mat,De_V_Mat,De_I_Mat,c_num);%1
            results_set(1)=results;
            if results~=0
                 Z_loop_update(results);
                for k=1:size(Z_circle.v_c,2)-1
                    var_increase= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{Z_circle.v_c(1,k),Z_circle.v_c(2,k)}==1&P_SVM.alpha_i_yi==0);
                    grad_increase=P_SVM.grad(var_increase);
                    [B,I]=min(grad_increase);
                    if ~isempty(B)
                        In_V_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=B;
                        In_I_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=var_increase(I);
                    else
                        In_V_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=NaN;
                        In_I_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=NaN;
                    end
                end
				k=size(Z_circle.v_c,2);
				var_decrease= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{Z_circle.v_c(1,k),Z_circle.v_c(2,k)}==1&P_SVM.alpha_i_yi==0);
                    grad_decrease=P_SVM.grad(var_decrease);
                    [B,I]=max(grad_decrease);
                    if ~isempty(B)
                        De_V_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=B;
                        De_I_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=var_decrease(I);
                    else
                        De_V_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=NaN;
                        De_I_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=NaN;
                    end
                In_V_Mat(~isnan(In_V_Mat))=P_SVM.grad(In_I_Mat(~isnan(In_V_Mat)));
            end
            end
            if De_V_Mat(i,j)>0.001
            De_V_Mat(~isnan(De_V_Mat))=P_SVM.grad(De_I_Mat(~isnan(De_V_Mat)));
            results=search_Z_loop_vertex_de(1,c_num,i,j,In_V_Mat,In_I_Mat,De_V_Mat,De_I_Mat,c_num);%3
            results_set(2)=results;
            if results~=0
                 Z_loop_update(results);
                for k=1:size(Z_circle.v_c,2)-1
                    var_decrease= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{Z_circle.v_c(1,k),Z_circle.v_c(2,k)}==1&P_SVM.alpha_i_yi==0);
                    grad_decrease=P_SVM.grad(var_decrease);
                    [B,I]=max(grad_decrease);
                    if ~isempty(B)
                        De_V_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=B;
                        De_I_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=var_decrease(I);
                    else
                        De_V_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=NaN;
                        De_I_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=NaN;
                    end
                end
				k=size(Z_circle.v_c,2);
				var_increase= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{Z_circle.v_c(1,k),Z_circle.v_c(2,k)}==1&P_SVM.alpha_i_yi==0);
                    grad_increase=P_SVM.grad(var_increase);
                    [B,I]=min(grad_increase);
                    if ~isempty(B)
                        In_V_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=B;
                        In_I_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=var_increase(I);
                    else
                        In_V_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=NaN;
                        In_I_Mat(Z_circle.v_c(1,k),Z_circle.v_c(2,k))=NaN;
                    end
				In_V_Mat(~isnan(In_V_Mat))=P_SVM.grad(In_I_Mat(~isnan(In_V_Mat)));
                De_V_Mat(~isnan(De_V_Mat))=P_SVM.grad(De_I_Mat(~isnan(De_V_Mat)));
            end
            end
        end
    end
    iter4=iter4+1;
    if iter4>i_n(4)||sum(results_set)==0
        break;
    end
end
iter=iter+1;
fval_c(iter)=object_F(P_SVM.alpha);
% disp([iter,fval_c]);
if sum(abs(alpha_buffer-P_SVM.alpha))<0.00001||iter>1000
    break;
end
end
if nargout==2
    fval=fval_c;
end
model.alpha=P_SVM.alpha;
P_SVM.alpha;
% sub_function 1
function [j2,l2]=search_vertex(c_min,c_max,i1,k1,sw)
    j2=0;
    l2=0;
    loop_convergence_ind=0;
    if sw==1
    for j1=c_min:c_max
    if loop_convergence_ind==1
        break;
    end
    for l1=c_min:c_max
        if loop_convergence_ind==1
            break;
        end
        if ~isempty(find(nested_rectangular_circle.v_c(1,:)==j1, 1))||~isempty(find(nested_rectangular_circle.v_c(2,:)==l1, 1))
           continue; 
        end
        grad_ds=zeros(1,4);
        grad_Is=zeros(1,4);
        var_increase1s= find(P_SVM.ntpT{i1,k1}==1);
        var_decrease2s= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{j1,k1}==1&P_SVM.alpha_i_yi==0);
        var_increase3s= find(P_SVM.ntpT{j1,l1}==1);
        var_decrease4s= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{i1,l1}==1&P_SVM.alpha_i_yi==0);
        if isempty(var_increase1s)||isempty(var_decrease2s)||isempty(var_increase3s)||isempty(var_decrease4s)
            continue;
        end
        [temp_Bs,temp_Is]=min(P_SVM.grad(var_increase1s));
        grad_ds(1,1)=temp_Bs;
        grad_Is(1,1)=var_increase1s(temp_Is);
        [temp_Bs,temp_Is]=max(P_SVM.grad(var_decrease2s));
        grad_ds(1,2)=-temp_Bs;
        grad_Is(1,2)=var_decrease2s(temp_Is);
        [temp_Bs,temp_Is_]=min(P_SVM.grad(var_increase3s));
        grad_ds(1,3)=temp_Bs;
        grad_Is(1,3)=var_increase3s(temp_Is_);
        [temp_Bs,temp_Is]=max(P_SVM.grad(var_decrease4s));
        grad_ds(1,4)=-temp_Bs;
        grad_Is(1,4)=var_decrease4s(temp_Is);
        if P_SVM.alpha(var_increase3s(temp_Is_))<P_SVM.C-P_SVM.epsilon&&P_SVM.alpha_i_yi(var_increase3s(temp_Is_))==0
            grad_loops=sum([nested_rectangular_circle.grad,grad_ds(2:end)]);
        else
            grad_loops=sum([nested_rectangular_circle.grad,grad_ds([2,4])]);
        end
        if grad_loops<-P_SVM.tol(3)
%             deta_ups=P_SVM.C-P_SVM.alpha(grad_Is(3));
            deta_lows=P_SVM.alpha(grad_Is([2,4]));
            distances=deta_lows;
            nested_rectangular_circle.grad=[nested_rectangular_circle.grad,grad_ds([2,4])];
            nested_rectangular_circle.v=[nested_rectangular_circle.v,grad_Is([2,4])];
            nested_rectangular_circle.v_c=[nested_rectangular_circle.v_c,[j1;l1]];
            nested_rectangular_circle.ind=[nested_rectangular_circle.ind,[-1,-1]];
            nested_rectangular_circle.step=min([distances,nested_rectangular_circle.step]);
            j2=j1;
            l2=l1;
            loop_convergence_ind=1;
        end
    end
    end
    else
    for j1=c_min:c_max
    if loop_convergence_ind==1
        break;
    end
    for l1=c_min:c_max
        if loop_convergence_ind==1
            break;
        end
        if ~isempty(find(nested_rectangular_circle.v_c(1,:)==j1, 1))||~isempty(find(nested_rectangular_circle.v_c(2,:)==l1, 1))
           continue; 
        end
        grad_ds=zeros(1,4);
        grad_Is=zeros(1,4);
        var_decrease1s= find(P_SVM.ntpT{i1,k1}==1);
        var_increase2s= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{j1,k1}==1&P_SVM.alpha_i_yi==0);
        var_decrease3s= find(P_SVM.ntpT{j1,l1}==1);
        var_increase4s= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{i1,l1}==1&P_SVM.alpha_i_yi==0);
        if isempty(var_decrease1s)||isempty(var_increase2s)||isempty(var_decrease3s)||isempty(var_increase4s)
            continue;
        end
        [temp_Bs,temp_Is]=max(P_SVM.grad(var_decrease1s));
        grad_ds(1,1)=-temp_Bs;
        grad_Is(1,1)=var_decrease1s(temp_Is);
        [temp_Bs,temp_Is]=min(P_SVM.grad(var_increase2s));
        grad_ds(1,2)=temp_Bs;
        grad_Is(1,2)=var_increase2s(temp_Is);
        [temp_Bs,temp_Is_]=max(P_SVM.grad(var_decrease3s));
        grad_ds(1,3)=-temp_Bs;
        grad_Is(1,3)=var_decrease3s(temp_Is_);
        [temp_Bs,temp_Is]=min(P_SVM.grad(var_increase4s));
        grad_ds(1,4)=temp_Bs;
        grad_Is(1,4)=var_increase4s(temp_Is);
        if P_SVM.alpha(var_decrease3s(temp_Is_))>P_SVM.epsilon&&P_SVM.alpha_i_yi(var_decrease3s(temp_Is_))==0
            grad_loops=sum([nested_rectangular_circle.grad,grad_ds(2:end)]);
        else
            grad_loops=sum([nested_rectangular_circle.grad,grad_ds([2,4])]);
        end
        if grad_loops<-P_SVM.tol(3)
            deta_ups=P_SVM.C-P_SVM.alpha(grad_Is([2,4]));
%             deta_lows=P_SVM.alpha(grad_Is(3));
            distances=deta_ups;
            nested_rectangular_circle.grad=[nested_rectangular_circle.grad,grad_ds([2,4])];
            nested_rectangular_circle.v=[nested_rectangular_circle.v,grad_Is([2,4])];
            nested_rectangular_circle.v_c=[nested_rectangular_circle.v_c,[j1;l1]];
            nested_rectangular_circle.ind=[nested_rectangular_circle.ind,[1,1]];
            nested_rectangular_circle.step=min([distances,nested_rectangular_circle.step]);
            j2=j1;
            l2=l1;
            loop_convergence_ind=1;
        end
    end
    end
    end
end
%sub_function 2
function [j2,l2]=search_first_vertex(c_min,c_max,i1,k1,sw)
    j2=0;
    l2=0;
    loop_convergence_ind=0;
    if sw==1
    for j1=c_min:c_max
    if loop_convergence_ind==1
        break;
    end
    for l1=c_min:c_max
        if loop_convergence_ind==1
            break;
        end
        if ~isempty(find(nested_rectangular_circle.v_c(1,:)==j1, 1))||~isempty(find(nested_rectangular_circle.v_c(2,:)==l1, 1))
           continue; 
        end
        var_increase1s= find(P_SVM.ntpT{i1,k1}==1);
        var_decrease2s= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{j1,k1}==1&P_SVM.alpha_i_yi==0);
        var_increase3s= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{j1,l1}==1&P_SVM.alpha_i_yi==0);
        var_decrease4s= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{i1,l1}==1&P_SVM.alpha_i_yi==0);        
        if isempty(var_increase1s)||isempty(var_decrease2s)||isempty(var_increase3s)||isempty(var_decrease4s)
            continue;
        end
%         disp('~isempty');
        grad_ds=zeros(1,4);
        grad_Is=zeros(1,4);
        [temp_Bs,temp_Is]=min(P_SVM.grad(var_increase1s));
        grad_ds(1,1)=temp_Bs;
        grad_Is(1,1)=var_increase1s(temp_Is);
        [temp_Bs,temp_Is]=max(P_SVM.grad(var_decrease2s));
        grad_ds(1,2)=-temp_Bs;
        grad_Is(1,2)=var_decrease2s(temp_Is);
        [temp_Bs,temp_Is]=min(P_SVM.grad(var_increase3s));
        grad_ds(1,3)=temp_Bs;
        grad_Is(1,3)=var_increase3s(temp_Is);
        [temp_Bs,temp_Is]=max(P_SVM.grad(var_decrease4s));
        grad_ds(1,4)=-temp_Bs;
        grad_Is(1,4)=var_decrease4s(temp_Is);
        grad_loops=sum([nested_rectangular_circle.grad,grad_ds(2:end)]);
        if grad_loops<-P_SVM.tol(3)
            deta_ups=P_SVM.C-P_SVM.alpha(grad_Is(3));
            deta_lows=P_SVM.alpha(grad_Is([2,4]));
            distances=min([deta_ups,deta_lows]);
            nested_rectangular_circle.grad=[nested_rectangular_circle.grad,grad_ds(2:end)];
            nested_rectangular_circle.v=[nested_rectangular_circle.v,grad_Is(2:end)];
            nested_rectangular_circle.v_c=[nested_rectangular_circle.v_c,[j1;l1]];
            nested_rectangular_circle.ind=[nested_rectangular_circle.ind,[-1,1,-1]];
            nested_rectangular_circle.step=min([distances,nested_rectangular_circle.step]);
            j2=j1;
            l2=l1;
            loop_convergence_ind=1;
        end
    end
    end
    else
    for j1=c_min:c_max
    if loop_convergence_ind==1
        break;
    end
    for l1=c_min:c_max
        if loop_convergence_ind==1
            break;
        end
        if ~isempty(find(nested_rectangular_circle.v_c(1,:)==j1, 1))||~isempty(find(nested_rectangular_circle.v_c(2,:)==l1, 1))
           continue; 
        end
        var_decrease1s= find(P_SVM.ntpT{i1,k1}==1);
        var_increase2s= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{j1,k1}==1&P_SVM.alpha_i_yi==0);
        var_decrease3s= find(P_SVM.alpha>P_SVM.epsilon&P_SVM.ntpT{j1,l1}==1&P_SVM.alpha_i_yi==0);
        var_increase4s= find(P_SVM.alpha<P_SVM.C-P_SVM.epsilon&P_SVM.ntpT{i1,l1}==1&P_SVM.alpha_i_yi==0);
        if isempty(var_decrease1s)||isempty(var_increase2s)||isempty(var_decrease3s)||isempty(var_increase4s)
            continue;
        end
%         disp('~isempty');
        grad_ds=zeros(1,4);
        grad_Is=zeros(1,4);
        [temp_Bs,temp_Is]=max(P_SVM.grad(var_decrease1s));
        grad_ds(1,1)=-temp_Bs;
        grad_Is(1,1)=var_decrease1s(temp_Is);
        [temp_Bs,temp_Is]=min(P_SVM.grad(var_increase2s));
        grad_ds(1,2)=temp_Bs;
        grad_Is(1,2)=var_increase2s(temp_Is);
        [temp_Bs,temp_Is]=max(P_SVM.grad(var_decrease3s));
        grad_ds(1,3)=-temp_Bs;
        grad_Is(1,3)=var_decrease3s(temp_Is);
        [temp_Bs,temp_Is]=min(P_SVM.grad(var_increase4s));
        grad_ds(1,4)=temp_Bs;
        grad_Is(1,4)=var_increase4s(temp_Is);
        grad_loops=sum([nested_rectangular_circle.grad,grad_ds(2:end)]);
        if grad_loops<-P_SVM.tol(3)
            deta_ups=P_SVM.C-P_SVM.alpha(grad_Is([2,4]));
            deta_lows=P_SVM.alpha(grad_Is(3));
            distances=min([deta_ups,deta_lows]);
            nested_rectangular_circle.grad=[nested_rectangular_circle.grad,grad_ds(2:end)];
            nested_rectangular_circle.v=[nested_rectangular_circle.v,grad_Is(2:end)];
            nested_rectangular_circle.v_c=[nested_rectangular_circle.v_c,[j1;l1]];
            nested_rectangular_circle.ind=[nested_rectangular_circle.ind,[1,-1,1]];
            nested_rectangular_circle.step=min([distances,nested_rectangular_circle.step]);
            j2=j1;
            l2=l1;
            loop_convergence_ind=1;
        end
    end
    end
    end
end
%sub function 3
    function multiple_loop_update()
        step_optimal_values=-sum(nested_rectangular_circle.grad)/(2*nested_rectangular_circle.ind*P_SVM.Kcache([nested_rectangular_circle.v],[nested_rectangular_circle.v])*nested_rectangular_circle.ind');
            step_values=min([step_optimal_values,nested_rectangular_circle.step]);
            P_SVM.alpha(nested_rectangular_circle.v(nested_rectangular_circle.ind==1))=P_SVM.alpha(nested_rectangular_circle.v(nested_rectangular_circle.ind==1))+step_values;
            P_SVM.alpha(nested_rectangular_circle.v(nested_rectangular_circle.ind==-1))=P_SVM.alpha(nested_rectangular_circle.v(nested_rectangular_circle.ind==-1))-step_values;
            P_SVM.grad=P_SVM.grad+2*P_SVM.Kcache(:,nested_rectangular_circle.v)*[step_values*nested_rectangular_circle.ind]';%updata gradient of object function
    end
% sub_function 4
function results=search_Z_loop_vertex_in(c_min,c_max,i1,k1,In_V_Mat,In_I_Mat,De_V_Mat,De_I_Mat,sw2)
    results=0;
    temp_c=c_min:c_max;
    temp_c2=c_min:c_max;
    for i_sw=1:sw2
        if i_sw==1
            temp_1=In_V_Mat(k1,c_min:c_max);
            if In_V_Mat(i1,k1)+temp_1(temp_c==i1)<-P_SVM.tol(4)
                Z_circle.grad=In_V_Mat(i1,k1)+temp_1(temp_c==i1);
                Z_circle.v_c=[[i1;k1],[k1;i1]];
                Z_circle.v=[In_I_Mat(i1,k1),[In_I_Mat(k1,i1)]];
                Z_circle.step=[P_SVM.C-P_SVM.alpha(In_I_Mat(i1,k1)),P_SVM.C-P_SVM.alpha(In_I_Mat(k1,i1))];
                results=1;
                break;
            end
                [I_temp_3]=find((In_V_Mat(i1,k1)+temp_1)<-P_SVM.tol(4));
                if isempty(I_temp_3)
                    results=0;
                    break;
                end
                    loc_rand=ceil(rand(1)*size(I_temp_3(:),1));
                    Z_circle.grad=In_V_Mat(i1,k1)+temp_1(I_temp_3(loc_rand));
                    Z_circle.v_c=[[i1;k1],[k1;temp_c(I_temp_3(loc_rand))]];
                    Z_circle.v=[In_I_Mat(i1,k1),[In_I_Mat(k1,temp_c(I_temp_3(loc_rand)))]];
                    Z_circle.step=[P_SVM.C-P_SVM.alpha(In_I_Mat(i1,k1)),P_SVM.C-P_SVM.alpha(In_I_Mat(k1,temp_c(I_temp_3(loc_rand))))];
                    temp_c2(k1)=0;
        else            
        temp_1=In_V_Mat(Z_circle.v_c(2,end),c_min:c_max);
        temp_1(temp_c2==0)=NaN;
            if Z_circle.grad+temp_1(temp_c==i1)<-P_SVM.tol(4)
                temp_num1=size(find(Z_circle.v==In_I_Mat(Z_circle.v_c(2,end),i1)),2)+1;
                Z_circle.grad=Z_circle.grad+temp_1(temp_c==i1);
                Z_circle.v_c=[Z_circle.v_c,[Z_circle.v_c(2,end);i1]];
                Z_circle.v=[Z_circle.v,[In_I_Mat(Z_circle.v_c(2,end-1),i1)]];
                Z_circle.step=[Z_circle.step,(P_SVM.C-P_SVM.alpha(In_I_Mat(Z_circle.v_c(2,end-1),i1)))/temp_num1];
                results=1;
                break;
            end
		temp_De=De_V_Mat(i1,Z_circle.v_c(2,end));
            if Z_circle.grad-temp_De<-P_SVM.tol(4)
                %temp_num1=size(find(Z_circle.v==In_I_Mat(Z_circle.v_c(2,end),i1)),2)+1;
                Z_circle.grad=Z_circle.grad-temp_De;
                Z_circle.v_c=[Z_circle.v_c,[i1;Z_circle.v_c(2,end)]];
                Z_circle.v=[Z_circle.v,[De_I_Mat(i1,Z_circle.v_c(2,end-1))]];
                Z_circle.step=[Z_circle.step,(P_SVM.alpha(De_I_Mat(i1,Z_circle.v_c(2,end-1))))];
                results=2;
                break;
            end	
                [I_temp_3]=find((Z_circle.grad+temp_1)<-P_SVM.tol(4));
                if isempty(I_temp_3)
                    results=0;
                    break;
                end
                    loc_rand=ceil(rand(1)*size(I_temp_3(:),1));
                    Z_circle.grad=Z_circle.grad+temp_1(temp_c(I_temp_3(loc_rand)));
                    Z_circle.v_c=[Z_circle.v_c,[Z_circle.v_c(2,end);temp_c(I_temp_3(loc_rand))]];
                    Z_circle.v=[Z_circle.v,[In_I_Mat(Z_circle.v_c(2,end-1),temp_c(I_temp_3(loc_rand)))]];
                    Z_circle.step=[Z_circle.step,P_SVM.C-P_SVM.alpha(In_I_Mat(Z_circle.v_c(2,end-1),temp_c(I_temp_3(loc_rand))))];
                    temp_c2(temp_c(I_temp_3(loc_rand)))=0;
        end
    end
end
% sub_function 5
function results=search_Z_loop_vertex_de(c_min,c_max,i1,k1,In_V_Mat,In_I_Mat,De_V_Mat,De_I_Mat,sw2)
    results=0;
    temp_c=c_min:c_max;
    temp_c2=c_min:c_max;
    for i_sw=1:sw2
        if i_sw==1
            temp_1=De_V_Mat(k1,c_min:c_max);
            if De_V_Mat(i1,k1)+temp_1(temp_c==i1)>P_SVM.tol(4)
                Z_circle.grad=De_V_Mat(i1,k1)+temp_1(temp_c==i1);
                Z_circle.v_c=[[i1;k1],[k1;i1]];
                Z_circle.v=[De_I_Mat(i1,k1),[De_I_Mat(k1,i1)]];
                Z_circle.step=[P_SVM.alpha(De_I_Mat(i1,k1)),P_SVM.alpha(De_I_Mat(k1,i1))];
                results=1;
                break;
            end
                [I_temp_3]=find((De_V_Mat(i1,k1)+temp_1)>P_SVM.tol(4));
                if isempty(I_temp_3)
                    results=0;
                    break;
                end
                    loc_rand=ceil(rand(1)*size(I_temp_3(:),1));
                    Z_circle.grad=De_V_Mat(i1,k1)+temp_1(I_temp_3(loc_rand));
                    Z_circle.v_c=[[i1;k1],[k1;temp_c(I_temp_3(loc_rand))]];
                    Z_circle.v=[De_I_Mat(i1,k1),[De_I_Mat(k1,temp_c(I_temp_3(loc_rand)))]];
                    Z_circle.step=[P_SVM.alpha(De_I_Mat(i1,k1)),P_SVM.alpha(De_I_Mat(k1,temp_c(I_temp_3(loc_rand))))];
                    temp_c2(k1)=0;
        else            
        temp_1=De_V_Mat(Z_circle.v_c(2,end),c_min:c_max);
        temp_1(temp_c2==0)=NaN;
            if Z_circle.grad+temp_1(temp_c==i1)>P_SVM.tol(4)
                Z_circle.grad=Z_circle.grad+temp_1(temp_c==i1);
                Z_circle.v_c=[Z_circle.v_c,[Z_circle.v_c(2,end);i1]];
                Z_circle.v=[Z_circle.v,[De_I_Mat(Z_circle.v_c(2,end-1),i1)]];
                Z_circle.step=[Z_circle.step,P_SVM.alpha(De_I_Mat(Z_circle.v_c(2,end-1),i1))];
                results=1;
                break;
            end
			temp_In=In_V_Mat(i1,Z_circle.v_c(2,end));
            if Z_circle.grad-temp_In>P_SVM.tol(4)
                %temp_num1=size(find(Z_circle.v==In_I_Mat(Z_circle.v_c(2,end),i1)),2)+1;
                Z_circle.grad=Z_circle.grad-temp_In;
                Z_circle.v_c=[Z_circle.v_c,[i1;Z_circle.v_c(2,end)]];
                Z_circle.v=[Z_circle.v,[In_I_Mat(i1,Z_circle.v_c(2,end-1))]];
                Z_circle.step=[Z_circle.step,(P_SVM.C-P_SVM.alpha(In_I_Mat(i1,Z_circle.v_c(2,end-1))))];
                results=2;
                break;
            end
                [I_temp_3]=find((Z_circle.grad+temp_1)>P_SVM.tol(4));
                if isempty(I_temp_3)
                    results=0;
                    break;
                end
                    loc_rand=ceil(rand(1)*size(I_temp_3(:),1));
                    temp_num1=size(find(Z_circle.v==De_I_Mat(Z_circle.v_c(2,end),I_temp_3(loc_rand))),2)+1;
                    Z_circle.grad=Z_circle.grad+temp_1(I_temp_3(loc_rand));
                    Z_circle.v_c=[Z_circle.v_c,[Z_circle.v_c(2,end);temp_c(I_temp_3(loc_rand))]];
                    Z_circle.v=[Z_circle.v,[De_I_Mat(Z_circle.v_c(2,end-1),temp_c(I_temp_3(loc_rand)))]];
                    Z_circle.step=[Z_circle.step,(P_SVM.alpha(De_I_Mat(Z_circle.v_c(2,end-1),temp_c(I_temp_3(loc_rand)))))/temp_num1];
                    temp_c2(temp_c(I_temp_3(loc_rand)))=0;
        end
    end
end
%sub function 6
    function Z_loop_update(results)
	if results==1
        if Z_circle.grad<0
        step_optimal_values=-Z_circle.grad/(2*ones(1,size(Z_circle.v,2))*P_SVM.Kcache([Z_circle.v],[Z_circle.v])*ones(1,size(Z_circle.v,2))');
            step_values=min([step_optimal_values,Z_circle.step]);
                P_SVM.alpha(Z_circle.v)=P_SVM.alpha(Z_circle.v)+step_values;
            P_SVM.grad=P_SVM.grad+2*P_SVM.Kcache(:,Z_circle.v)*[step_values*ones(1,size(Z_circle.v,2))]';%updata gradient of object function
        else
             step_optimal_values=Z_circle.grad/(2*ones(1,size(Z_circle.v,2))*P_SVM.Kcache([Z_circle.v],[Z_circle.v])*ones(1,size(Z_circle.v,2))');
            step_values=min([step_optimal_values,Z_circle.step]);
                P_SVM.alpha(Z_circle.v)=P_SVM.alpha(Z_circle.v)-step_values;
            P_SVM.grad=P_SVM.grad+2*P_SVM.Kcache(:,Z_circle.v)*(-step_values*ones(1,size(Z_circle.v,2)))';%updata gradient of object function
        end
	else
		if Z_circle.grad<0
        step_optimal_values=-Z_circle.grad/(2*[ones(1,size(Z_circle.v,2)-1),-1]*P_SVM.Kcache([Z_circle.v],[Z_circle.v])*[ones(1,size(Z_circle.v,2)-1),-1]');
            step_values=min([step_optimal_values,Z_circle.step]);
                P_SVM.alpha(Z_circle.v)=P_SVM.alpha(Z_circle.v)+step_values*[ones(1,size(Z_circle.v,2)-1),-1];
            P_SVM.grad=P_SVM.grad+2*P_SVM.Kcache(:,Z_circle.v)*[step_values*[ones(1,size(Z_circle.v,2)-1),-1]]';%updata gradient of object function
        else
             step_optimal_values=Z_circle.grad/(2*[ones(1,size(Z_circle.v,2)-1),-1]*P_SVM.Kcache([Z_circle.v],[Z_circle.v])*[ones(1,size(Z_circle.v,2)-1),-1]');
            step_values=min([step_optimal_values,Z_circle.step]);
                P_SVM.alpha(Z_circle.v)=P_SVM.alpha(Z_circle.v)-step_values*[ones(1,size(Z_circle.v,2)-1),-1];
            P_SVM.grad=P_SVM.grad+2*P_SVM.Kcache(:,Z_circle.v)*(-step_values*[ones(1,size(Z_circle.v,2)-1),-1])';%updata gradient of object function
        end
	end
    end
end

