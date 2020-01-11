rm(list=ls())
setwd("d:\\Documents\\wuhaoyu\\����\\daguan\\data")
library(kknn)
library(psych)
library(pROC) 
# ��������
train_data = read.csv("pfm_train.csv")
test_data = read.csv("pfm_test.csv")
train_Y = train_data['Attrition']
train_X = train_data[-grep('Attrition', colnames(train_data))]
all_X = rbind(train_X, test_data)
names(all_X)[1]='Age'
all_X = all_X[,-c(17,22)]
# ID�� EmployeeNumber
ID = all_X$EmployeeNumbe
# ��ֵ���ͱ���11��
num_var = c("Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked", "PercentSalaryHike",
            "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
            "YearsSinceLastPromotion", "YearsWithCurrManager")
num_X = all_X[,which(names(all_X) %in% num_var)]
# �������ͱ���10��
ord_var = c("BusinessTravel", "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel",
            "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel",
            "WorkLifeBalance")
ord_X = all_X[,which(names(all_X) %in% ord_var)]
ord_X$BusinessTravel = as.character(ord_X$BusinessTravel)
ord_X$BusinessTravel[ord_X$BusinessTravel=='Non-Travel'] = 0
ord_X$BusinessTravel[ord_X$BusinessTravel=='Travel_Rarely'] = 1
ord_X$BusinessTravel[ord_X$BusinessTravel=='Travel_Frequently'] = 2
ord_X$BusinessTravel = as.integer(ord_X$BusinessTravel)
# �������6��
chr_var = c("Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime")
chr_X = all_X[,which(names(all_X) %in% chr_var)]
chr_dummies = dummyVars(~Department+EducationField+Gender+JobRole+MaritalStatus+OverTime, data=chr_X)
chr_X_ohe = as.data.frame(predict(chr_dummies, newdata=chr_X))
# ���ȫ�����ݼ�
all_X_combined = cbind(ID,num_X, ord_X, chr_X_ohe)
# write.csv(all_X_combined, "all_X_combined.csv", row.names = FALSE)
# ���������ݼ�
#����Ԥ����
train_df = all_X_combined[which(all_X_combined$ID%in%train_data$EmployeeNumber),]
train_df = cbind(train_df,train_Y)
select = sample(1:nrow(train_df), length(train_df$ID)*0.7, replace=FALSE)
train_df = train_df[,-1]
train_df$Attrition = factor(train_df$Attrition)
test_df = train_df[-select,]
train_df = train_df[select,]

knn_model <- kknn(Attrition ~.,
                  train_df,
                  test_df,
                  k=7,
                  distance = 2)
#�ڲ��Լ���Ԥ��
pre_knn <- fitted(knn_model)
#�����������
table(test_df$Attrition, pre_knn,dnn=c("��ʵֵ","Ԥ��ֵ"))
#����ROC���߲�����AUCֵ
knn_roc <- roc(test_df$Attrition,as.numeric(pre_knn))
plot(knn_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='knn�㷨ROC����')
