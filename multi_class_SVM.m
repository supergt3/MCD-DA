function [model,b]=multi_class_SVM(X,Label,C)

[~,length]=size(X);
c_num=max(Label);
X_c=kron(X,ones(1,c_num));
alpha_num=length*c_num;
K_mat=zeros(alpha_num,alpha_num);
for k=1:c_num
   temp=zeros(1,length);
   temp(Label==k)=1;
   temp_vec=kron(temp,ones(1,c_num));
   temp_mat1=diag(temp_vec);
   temp=zeros(1,c_num);
   temp(k)=1;
   temp_vec=kron(ones(1,length),temp);
   temp_mat2=diag(temp_vec);
   X_m_1=0.5*temp_mat1*(X_c')*X_c*temp_mat1;
   X_m_2=0.5*temp_mat2*(X_c')*X_c*temp_mat2;
   X_m_31=-0.5*temp_mat1*(X_c')*X_c*temp_mat2;
   X_m_32=-0.5*temp_mat2*(X_c')*X_c*temp_mat1;
   K_mat=K_mat+(X_m_1+X_m_2+X_m_31+X_m_32);
end
 tic
 alpha_i_yi=zeros(c_num,length);
    for i=1:c_num
        temp=zeros(1,length);
        temp(Label==i)=1;
        alpha_i_yi(i,:)=temp;
    end
    alpha_i_yi=alpha_i_yi(:)';
%     alpha_init = zeros(1,alpha_num);
%     alpha_init(alpha_i_yi==1)=0;
    Aeq1=zeros(c_num,alpha_num);
    Aeq2=zeros(c_num,alpha_num);
    for i=1:c_num
        temp=zeros(1,length);
        temp(Label==i)=1;
        temp_vec=kron(temp,ones(1,c_num));
        Aeq1(i,:)=temp_vec;
         temp=zeros(1,c_num);
         temp(i)=1;
         temp_vec=kron(ones(1,length),temp);
         Aeq2(i,:)=temp_vec;
    end
    Aeq=Aeq1-Aeq2;
    C_f=C*ones(1,alpha_num);
    C_f(alpha_i_yi==1)=0;
    object_F=@(alpha_v) alpha_v*K_mat*alpha_v'-2*alpha_v*ones(size(alpha_v,2),1);
%%
%     options = optimoptions('fmincon','Algorithm','interior-point','Display','iter','OptimalityTolerance',10^-12,'MaxIterations',1000000,'MaxFunctionEvaluations',1000000);%'Display','iter', 'sqp','MaxIterations',1000000,'MaxFunctionEvaluations',1000000
%    options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','iter','OptimalityTolerance',10^-12,'MaxIterations',1000);%'Display','iter', 'sqp','MaxIterations',1000000,'MaxFunctionEvaluations',1000000
% %   [alpha,fval,exitflag]=fmincon(object_F,alpha_init,[],[],Aeq,zeros(c_num,1),zeros(1,alpha_num),C_f,[],options);
%   [alpha,fval,exitflag]=quadprog(2*K_mat,-2*ones(1,alpha_num),[],[],Aeq,zeros(c_num,1),zeros(1,alpha_num),C_f,[],options);
%   toc
  tol=[0.001,0.001,0.001,0.001];
  tic
  Par=MCD_DA(K_mat, Label, C, tol,c_num);
  toc
  alpha=Par.alpha;
  object_F(alpha)
%   fval
for i=1:c_num
        X_temp=X_c;
        X_temp2=X_c;
       temp=zeros(1,length);
       temp(Label==i)=1;
       temp_vec1=kron(temp,ones(1,c_num));
       temp=zeros(1,c_num);
       temp(i)=1;
       temp_vec2=kron(ones(1,length),temp);
       temp=X_temp2*(temp_vec1.*alpha)'-X_temp*(temp_vec2.*alpha)';
       model(i).Proj=temp;
end
 %calculat b
 distance_yi_m=zeros(c_num,c_num);
distance_yi_m_num=zeros(c_num,c_num);
for i=1:c_num
   temp_L=find(Label==i);
   for j=1:size(temp_L(:),1)
       temp=zeros(1,length);
       temp(temp_L(j))=1;
       temp_vec1=kron(temp,ones(1,c_num));
       SV=find(alpha>0.0001&alpha<C-0.0001&temp_vec1==1);
        if ~isempty(SV)
           for k=1:size(SV(:),1)
                [c_ind,l_ind]=ind2sub([c_num,length],SV(k));
                temp_yi=X(:,l_ind)'*model(i).Proj;
                temp_m=X(:,l_ind)'*model(c_ind).Proj;
                distance_yi_m(i,c_ind)=distance_yi_m(i,c_ind)+temp_yi-temp_m;
                distance_yi_m_num(i,c_ind)=distance_yi_m_num(i,c_ind)+1;
           end
        end
   end  
end
bp_Mat=zeros(c_num*(c_num-1),c_num);
for i=1:c_num
    temp=eye(c_num-1);
    temp2=zeros(c_num-1,c_num);
    temp2(:,[1:i-1,i+1:c_num])=temp;
    temp2(:,i)=-1;
    bp_Mat((i-1)*(c_num-1) +1:i*(c_num-1),1:c_num)=temp2;
end
% distance_bp=zeros((c_num-1)*c_num,1);
temp=eye(c_num);
distance_yi_m_num=distance_yi_m_num';
distance_yi_m_num_vec=distance_yi_m_num(temp==0);
distance_yi_m=distance_yi_m';
distance_yi_m_vec=distance_yi_m(temp==0);
bp_Mat_SV=bp_Mat(distance_yi_m_num_vec~=0,:);
b_SV=distance_yi_m_vec(distance_yi_m_num_vec~=0)./distance_yi_m_num_vec(distance_yi_m_num_vec~=0)-2;%-2
if ~isempty(bp_Mat_SV)
    Aeq_SV=bp_Mat_SV;
    beq_SV=b_SV;
else
    Aeq_SV=[];
    beq_SV=[];
end
alpha
b=((Aeq_SV(:,1:c_num)'*Aeq_SV(:,1:c_num))+0.001*eye(c_num))^-1*Aeq_SV(:,1:c_num)'*beq_SV;
end