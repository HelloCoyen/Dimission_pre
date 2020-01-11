rm(list=ls())
setwd("d:\\Documents\\wuhaoyu\\桌面\\daguan\\data")
library(psych)
library(pROC)
library(PerformanceAnalytics)
library(BBmisc)
install.packages("BBmisc")

normalize()
# 加载数据
train_data = read.csv("pfm_train.csv")
test_data = read.csv("pfm_test.csv")
train_Y = train_data['Attrition']
train_X = train_data[-grep('Attrition', colnames(train_data))]
all_X = rbind(train_X, test_data)
names(all_X)[1]='Age'
all_X = all_X[,-c(17,22)]
# ID列 EmployeeNumber
ID = all_X$EmployeeNumbe
# 数值类型变量11个
num_var = c("Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked", "PercentSalaryHike",
            "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
            "YearsSinceLastPromotion", "YearsWithCurrManager")
num_X = all_X[,which(names(all_X) %in% num_var)]
# 定序类型变量10个
ord_var = c("BusinessTravel", "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel",
            "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel",
            "WorkLifeBalance")
ord_X = all_X[,which(names(all_X) %in% ord_var)]
ord_X$BusinessTravel = as.character(ord_X$BusinessTravel)
ord_X$BusinessTravel[ord_X$BusinessTravel=='Non-Travel'] = 0
ord_X$BusinessTravel[ord_X$BusinessTravel=='Travel_Rarely'] = 1
ord_X$BusinessTravel[ord_X$BusinessTravel=='Travel_Frequently'] = 2
ord_X$BusinessTravel = as.integer(ord_X$BusinessTravel)
# 分类变量6个
chr_var = c("Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime")
chr_X = all_X[,which(names(all_X) %in% chr_var)]
# 组合全量数据集
all_X_combined = cbind(ID,num_X, ord_X, chr_X)
# 相关系数
num_cor = cor(all_X_combined[num_var],method = "pearson")
chart.Correlation(all_X_combined[num_var])
ord_cor = cor(all_X_combined[ord_var],method = "spearman")
chart.Correlation(all_X_combined[ord_cor])
chr_cor = cor(all_X_combined[chr_var],method = "kendall")
# 数据处理
# 组合全量数据集
all_X_combined$Age = log(all_X_combined$Age)
all_X_combined$DistanceFromHome = log(all_X_combined$DistanceFromHome)
all_X_combined$MonthlyIncome = log10(all_X_combined$MonthlyIncome)
all_X_combined$PercentSalaryHike = log(all_X_combined$PercentSalaryHike)
all_X_combined$TotalWorkingYears = all_X_combined$TotalWorkingYears^0.5
all_X_combined$YearsAtCompany = all_X_combined$YearsAtCompany^(0.5)
all_X_combined$YearsInCurrentRole = all_X_combined$YearsInCurrentRole^(0.5)
all_X_combined$YearsSinceLastPromotion = all_X_combined$YearsSinceLastPromotion^(0.5)
all_X_combined$YearsWithCurrManager = all_X_combined$YearsWithCurrManager^(0.5)

all_X_combined[ord_var] = lapply(all_X_combined[ord_var],function(x) as.factor(x))
describe(all_X_combined)
str(all_X_combined)

# 预测数据
train_df = all_X_combined[which(all_X_combined$ID%in%train_data$EmployeeNumber),]
train_df = cbind(train_df,train_Y)
train_df = train_df[,-1]
train_df$Attrition = factor(train_df$Attrition)
test_df = all_X_combined[which(all_X_combined$ID%in%test_data$EmployeeNumber),]
test_df = test_df[,-1]
logit_model <- glm(Attrition~.,family = binomial(link = "logit"),data = train_df)
logit_model.step <- step(logit_model,direction = 'both')

#在测试集上预测
result <-  predict(logit_model.step,test_df,interval="confidence")
result <- ifelse(result>0,1,0)
sum(result)
res = data.frame(result)

write.csv(res,"result_logit.csv",row.names = FALSE)
res$result <- as.integer(res$result)
(sum(res$result))/350
