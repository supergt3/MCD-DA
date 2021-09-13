X=rand(200,80);%generate samples
label=kron(ones(1,20),[1:4]);%generate label
C=1;
[model,b]=multi_class_SVM(X,label,C);%using MCD-DA train the multiclass SVM
[predict_label,accuracy]=multi_class_SVM_test(X,model,b,label);%test accuracy