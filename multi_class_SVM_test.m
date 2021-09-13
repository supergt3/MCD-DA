function [predict_c,accuracy]=multi_class_SVM_test(X,model,b,test_label)
predict_X=zeros(size(model,2),size(X,2));
for k=1:size(X,2)
    for i=1:size(model,2)
        predict_X(i,k)=X(:,k)'*model(i).Proj+b(i);
    end
end
[~,predict_c]=max(predict_X);
temp=predict_c(:)==test_label(:);
accuracy=size(find(temp(:)==1),1)/size(temp(:),1);
end