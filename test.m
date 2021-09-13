
%%
%test iris dataset
f=fopen('.\iris\iris.data');% open dataset
data=textscan(f,'%f,%f,%f,%f,%s'); %read data
Label_total=zeros(1,size(data{1},1));
class_name={'Iris-setosa','Iris-versicolor','Iris-virginica'};
for i=1:size(data{1},1)
    temp=strcmp(class_name,data{5}(i));
    temp2=find(temp==1);
    Label_total(i)=temp2;
end
data_=cell2mat(data(1:4));
data2=mapminmax(data_',0,1);
data2=data2';
elapsedTime=zeros(1,6);
fval_total=zeros(1,6);
fval_c=cell(1,5);
for j=1:5
X=[];
for i=1:size(class_name,2)
    temp=find(Label_total==i);
    X=[X,data2(temp(1:10*j),:)'];
end
C=1;
Label=kron([1:size(class_name,2)],ones(1,10*j));
[dim_x,length]=size(X);
c_num=max(Label);
X_c=kron(X,ones(1,c_num));
alpha_num=length*c_num;
K_mat=zeros(alpha_num,alpha_num);% construct Q matrix
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
 alpha_i_yi=zeros(c_num,length);
    for i=1:c_num
        temp=zeros(1,length);
        temp(Label==i)=1;
        alpha_i_yi(i,:)=temp;
    end
    alpha_i_yi=alpha_i_yi(:)';
    alpha_init = zeros(1,alpha_num);
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
  tol=[0.001,0.001,0.001,0.001];
  tic
  [Par,fval_c{j}]=MCD_DA(K_mat, Label, C, tol,c_num); %perform MCD-DA algorithm
  elapsedTime(j)=toc;
  alpha=Par.alpha;
  fval_total(j)=object_F(alpha);
end
%plot convergence curve
x=[1:1000];
figure(1);
plot(x(1:size(fval_c{1},2)+1),[0,fval_c{1}],'-*r',x(1:size(fval_c{2},2)+1),[0,fval_c{2}],':xc',x(1:size(fval_c{3},2)+1),[0,fval_c{3}],'-sb',x(1:size(fval_c{4},2)+1),[0,fval_c{4}],'--dg',x(1:size(fval_c{5},2)+1),[0,fval_c{5}],':dm','LineWidth',2,'Markersize',5);
xlabel('Iteration number \itk');
ylabel('Values of objective function');
legend('\itN\rm=30','\itN\rm=60','\itN\rm=90','\itN\rm=120','\itN\rm=150');
set(gca,'FontName','times new Roman','FontSize',12);
%%
%test dry_bean_dataset
[data,txt,raw]=xlsread('.\dry_bean\DryBeanDataset\Dry_Bean_Dataset.xlsx');
Label_total=zeros(1,size(data,1));
class_name={'SEKER','BARBUNYA','BOMBAY','CALI','HOROZ','SIRA','DERMASON'};
for i=1:size(data,1)
    temp=strcmp(class_name,raw(i+1,end));
    temp2=find(temp==1);
    Label_total(i)=temp2;
end
data2=mapminmax(data',0,1);
data2=data2';
elapsedTime=zeros(1,6);
fval_total=zeros(1,6);
fval_c=cell(1,6);
for j=2:6
X=[];
for i=1:size(class_name,2)
    temp=find(Label_total==i);
    X=[X,data2(temp(1:50*j),:)'];
end
C=1;
Label=kron([1:size(class_name,2)],ones(1,50*j));
[dim_x,length]=size(X);
c_num=max(Label);
X_c=kron(X,ones(1,c_num));
alpha_num=length*c_num;
K_mat=zeros(alpha_num,alpha_num);% construct Q matrix
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
 alpha_i_yi=zeros(c_num,length);
    for i=1:c_num
        temp=zeros(1,length);
        temp(Label==i)=1;
        alpha_i_yi(i,:)=temp;
    end
    alpha_i_yi=alpha_i_yi(:)';
    alpha_init = zeros(1,alpha_num);
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
  tol=[0.001,0.001,0.001,0.001];
  tic
  [Par,fval_c{j}]=MCD_DA(K_mat, Label, C, tol,c_num);%perform MCD-DA algorithm
  elapsedTime(j)=toc;
  alpha=Par.alpha;
  fval_total(j)=object_F(alpha);
end
%plot convergence curve
x=[1:1000];
figure(1);
plot(x(1:size(fval_c{2},2)+1),[0,fval_c{2}],'-*r',x(1:size(fval_c{3},2)+1),[0,fval_c{3}],':xc',x(1:size(fval_c{4},2)+1),[0,fval_c{4}],'-sb',x(1:size(fval_c{5},2)+1),[0,fval_c{5}],'--dg',x(1:size(fval_c{6},2)+1),[0,fval_c{6}],':dm','LineWidth',2,'Markersize',5);
xlabel('Iteration number \itk');
ylabel('Values of objective function');
legend('\itN\rm=700','\itN\rm=1050','\itN\rm=1400','\itN\rm=1750','\itN\rm=2100');
set(gca,'FontName','times new Roman','FontSize',12);