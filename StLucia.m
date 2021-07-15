
clc;clear
load pinsondata.mat
for iter=1:30
[idx,c]=kmeans(pinsondata(:,7:8),4);
 MIX=[pinsondata,idx];
pinsondataNEW=[pinsondata,idx];

  day=MIX(:,1);
  maxday=max(day);
 testnum=randperm(maxday,30);
 testnum=transpose(testnum);

output = [];
for i=1:30
    j=find(MIX(:,1)==testnum(i,1));
   
for k=1:size(j)
output=[output;j(k,1)];   
end

end
test=MIX(output(:,1),:);
ss=transpose(testnum);
for i=1:length(ss)
MIX=MIX(find(MIX(:,1)~=ss(i)),:);
 end
    pinsondataclus1=MIX(find(MIX(:,9)==1),:);
         pinsondataclus2=MIX(find(MIX(:,9)==2),:);
    pinsondataclus3=MIX(find(MIX(:,9)==3),:);
       pinsondataclus4=MIX(find(MIX(:,9)==4),:);

 
 trainingData1=pinsondataclus1;
 trainingData2=pinsondataclus2;
 trainingData3=pinsondataclus3;
 trainingData4=pinsondataclus4;
 [trainedclus1, validationRMSE1(:,iter)] = trainRegressionModel(trainingData1);
 [trainedclus2, validationRMSE2(:,iter)] = trainRegressionModel(trainingData2);
 [trainedclus3, validationRMSE3(:,iter)] = trainRegressionModel(trainingData3);
 [trainedclus4, validationRMSE4(:,iter)] = trainRegressionModel(trainingData4);
 xtot1=[pinsondataclus1(:,1:7)]
 ytot1 = trainedclus1.predictFcn(xtot1);
 A=[pinsondataclus1(:,8),ytot1];
MAE1(:,iter) = mean(abs(A(:,1) - A(:,2)));
 xtot2=[pinsondataclus2(:,1:7)]
 ytot2 = trainedclus2.predictFcn(xtot2);
 B=[pinsondataclus2(:,8),ytot2];
MAE2(:,iter) = mean(abs(B(:,1) - B(:,2)));

 xtot3=[pinsondataclus3(:,1:7)];
 ytot3 = trainedclus3.predictFcn(xtot3);
 C=[pinsondataclus3(:,8),ytot3];
MAE3(:,iter) = mean(abs(C(:,1) - C(:,2)));

 xtot4=[pinsondataclus4(:,1:7)];
 ytot4 = trainedclus4.predictFcn(xtot4);
 D=[pinsondataclus4(:,8),ytot4];
MAE4(:,iter) = mean(abs(D(:,1) - D(:,2)));

 mse1(:,iter) = resubLoss(trainedclus1.RegressionGP);
 mse2(:,iter) = resubLoss(trainedclus2.RegressionGP);
 mse3(:,iter) = resubLoss(trainedclus3.RegressionGP);
 mse4(:,iter)= resubLoss(trainedclus4.RegressionGP);
 
 
disp('     valRMSE      valMSE      valMAE      %valRMSE    %valMAE ')
[validationRMSE1,mse1,MAE1,(1-((433000-validationRMSE1)/433000))*100,(1-((433000-MAE1)/433000))*100
  validationRMSE2,mse2,MAE2,(1-((433000-validationRMSE2)/433000))*100,(1-((433000-MAE2)/433000))*100
  validationRMSE3,mse3,MAE3,(1-((433000-validationRMSE3)/433000))*100,(1-((433000-MAE3)/433000))*100
  validationRMSE4,mse4,MAE4,(1-((433000-validationRMSE4)/433000))*100,(1-((433000-MAE4)/433000))*100];

 X1=[test(find(test(:,9)==1),1:7)];
 X2=[test(find(test(:,9)==2),1:7)];
 X3=[test(find(test(:,9)==3),1:7)];
 X4=[test(find(test(:,9)==4),1:7)];
[yfit1,ysd1,yint1] = trainedclus1.predictFcn(X1);
 [yfit2,ysd2,yint2] = trainedclus2.predictFcn(X2);
 [yfit3,ysd3,yint3] = trainedclus3.predictFcn(X3);
 [yfit4,ysd4,yint4] = trainedclus4.predictFcn(X4);

 
ypred=[test(:,8:9),zeros(length(test),1)];
ypred(find(ypred(:,2)==1),3)=yfit1;
ypred(find(ypred(:,2)==2),3)=yfit2;
ypred(find(ypred(:,2)==3),3)=yfit3;
ypred(find(ypred(:,2)==4),3)=yfit4;
Yprtrue=[ypred(:,1),ypred(:,3),test(:,7)];



testRMSE(:,iter)=sqrt(sum((ypred(:,1)-ypred(:,3)).^2)/length(test));
testMSE(:,iter)=(sum((ypred(:,1)-ypred(:,3)).^2)/length(test));
testMAE(:,iter)=(sum(abs(((ypred(:,1)-ypred(:,3))))))/length(test);
%  epsil(:,iter)=Yprtrue(:,2)-Yprtrue(:,1);
end
