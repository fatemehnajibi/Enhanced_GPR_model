
clc;clear
load DEN.mat
% eva = evalclusters(DEN(:,11:12),'kmeans','gap','KList',[1:8])
for iter=1:30
[idx,c]=kmeans(DEN(:,11:12),4);
 MIX=[DEN,idx];
DENNEW=[DEN,idx];

  day=MIX(:,1);
  maxday=max(day);
  testnum=randperm(maxday,30);
 testnum=transpose(testnum);

  for i=1:30
      test(((i-1)*24+1):(24*i),:)=MIX((MIX(:,1)==testnum(i)),:);
  end
ss=transpose(testnum);
for i=1:length(ss)
MIX=MIX(MIX(:,1)~=ss(i),:);
end
    DENclus1=MIX(MIX(:,13)==1,:);
         DENclus2=MIX(MIX(:,13)==2,:);
    DENclus3=MIX(MIX(:,13)==3,:);
       DENclus4=MIX(MIX(:,13)==4,:);

 
 trainingData1=DENclus1;
 trainingData2=DENclus2;
 trainingData3=DENclus3;
 trainingData4=DENclus4;
 [trainedclus1, validationRMSE1(:,iter)] = trainRegressionModel(trainingData1);
 [trainedclus2, validationRMSE2(:,iter)] = trainRegressionModel(trainingData2);
 [trainedclus3, validationRMSE3(:,iter)] = trainRegressionModel(trainingData3);
 [trainedclus4, validationRMSE4(:,iter)] = trainRegressionModel(trainingData4);
 xtot1=[DENclus1(:,1:2),DENclus1(:,4:11)];
 ytot1 = trainedclus1.predictFcn(xtot1);
 A=[DENclus1(:,12),ytot1];
MAE1(:,iter) = mean(abs(A(:,1) - A(:,2)));
 xtot2=[DENclus2(:,1:2),DENclus2(:,4:11)];
 ytot2 = trainedclus2.predictFcn(xtot2);
 B=[DENclus2(:,12),ytot2];
MAE2(:,iter) = mean(abs(B(:,1) - B(:,2)));

 xtot3=[DENclus3(:,1:2),DENclus3(:,4:11)];
 ytot3 = trainedclus3.predictFcn(xtot3);
 C=[DENclus3(:,12),ytot3];
MAE3(:,iter)= mean(abs(C(:,1) - C(:,2)));

 xtot4=[DENclus4(:,1:2),DENclus4(:,4:11)];
 ytot4 = trainedclus4.predictFcn(xtot4);
 D=[DENclus4(:,12),ytot4];
MAE4(:,iter)= mean(abs(D(:,1) - D(:,2)));

 mse1(:,iter) = resubLoss(trainedclus1.RegressionGP);
 mse2(:,iter)= resubLoss(trainedclus2.RegressionGP);
 mse3(:,iter) = resubLoss(trainedclus3.RegressionGP);
 mse4(:,iter)= resubLoss(trainedclus4.RegressionGP);
 
 
disp('     valRMSE      valMSE      valMAE      %valRMSE    %valMAE ')
[validationRMSE1,mse1,MAE1,(1-((30-validationRMSE1)/30))*100,(1-((30-MAE1)/30))*100
  validationRMSE2,mse2,MAE2,(1-((30-validationRMSE2)/30))*100,(1-((30-MAE2)/30))*100
  validationRMSE3,mse3,MAE3,(1-((30-validationRMSE3)/30))*100,(1-((30-MAE3)/30))*100
  validationRMSE4,mse4,MAE4,(1-((30-validationRMSE4)/30))*100,(1-((30-MAE4)/30))*100];

 X1=[test(test(:,13)==1,1),test(find(test(:,13)==1),3:11)];
 X2=[test(test(:,13)==2,1),test(test(:,13)==2,3:11)];
 X3=[test(find(test(:,13)==3),1),test(find(test(:,13)==3),3:11)];
 X4=[test(find(test(:,13)==4),1),test(find(test(:,13)==4),3:11)];
[yfit1,ysd1,yint1] = trainedclus1.predictFcn(X11);
 [yfit2,ysd2,yint2] = trainedclus2.predictFcn(X22);
 [yfit3,ysd3,yint3] = trainedclus3.predictFcn(X33);
 [yfit4,ysd4,yint4] = trainedclus4.predictFcn(X44);

 
ypred=[test(:,12:13),zeros(720,1)];
ypred(find(ypred(:,2)==1),3)=yfit1;
ypred(ypred(:,2)==2,3)=yfit2;
ypred(find(ypred(:,2)==3),3)=yfit3;
ypred(find(ypred(:,2)==4),3)=yfit4;
Yprtrue=[ypred(:,1),ypred(:,3),DEN(1:720,11)]



testRMSE(:,iter)=sqrt(sum((ypred(:,1)-ypred(:,3)).^2)/720);
testMSE(:,iter)=(sum((ypred(:,1)-ypred(:,3)).^2)/720);
testMAE(:,iter)=(sum(abs(((ypred(:,1)-ypred(:,3))))))/720;

epsil(:,iter)=Yprtrue(:,2)-Yprtrue(:,1);
end 

\